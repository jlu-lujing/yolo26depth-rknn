//! RKNN C bindings and model wrapper.

use std::ffi::CString;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;

use log::info;

use crate::error::{Error, Result};

// ============================================================================
// C Types
// ============================================================================

#[allow(non_camel_case_types)]
type rknn_context = u64;

const RKNN_SUCC: c_int = 0;
const RKNN_QUERY_IN_OUT_NUM: c_uint = 0;
const RKNN_QUERY_INPUT_ATTR: c_uint = 1;
const RKNN_QUERY_OUTPUT_ATTR: c_uint = 2;
const RKNN_TENSOR_UINT8: c_uint = 3;
const RKNN_TENSOR_NHWC: c_uint = 1;
const RKNN_MAX_DIMS: usize = 16;
const RKNN_MAX_NAME_LEN: usize = 256;

#[repr(C)]
struct rknn_tensor_attr {
    index: c_uint,
    n_dims: c_uint,
    dims: [c_uint; RKNN_MAX_DIMS],
    name: [u8; RKNN_MAX_NAME_LEN],
    n_elems: c_uint,
    size: c_uint,
    fmt: c_uint,
    type_: c_uint,
    qnt_type: c_uint,
    fl: i8,
    zp: i32,
    scale: f32,
    w_stride: c_uint,
    size_with_stride: c_uint,
    pass_through: u8,
    h_stride: c_uint,
}

#[repr(C)]
struct rknn_input_output_num {
    n_input: c_uint,
    n_output: c_uint,
}

#[repr(C)]
struct rknn_input {
    index: c_uint,
    buf: *mut c_void,
    size: c_uint,
    pass_through: u8,
    type_: c_uint,
    fmt: c_uint,
}

#[repr(C)]
struct rknn_output {
    want_float: u8,
    is_prealloc: u8,
    index: c_uint,
    buf: *mut c_void,
    size: c_uint,
}

#[link(name = "rknnrt")]
extern "C" {
    fn rknn_init(
        context: *mut rknn_context,
        model: *const c_void,
        size: c_uint,
        flag: c_uint,
        extend: *const c_void,
    ) -> c_int;
    fn rknn_destroy(context: rknn_context) -> c_int;
    fn rknn_set_core_mask(context: rknn_context, core_mask: c_uint) -> c_int;
    fn rknn_query(
        context: rknn_context,
        cmd: c_uint,
        info: *mut c_void,
        size: c_uint,
    ) -> c_int;
    fn rknn_inputs_set(
        context: rknn_context,
        n_inputs: c_uint,
        inputs: *mut rknn_input,
    ) -> c_int;
    fn rknn_run(context: rknn_context, extend: *const c_void) -> c_int;
    fn rknn_outputs_get(
        context: rknn_context,
        n_outputs: c_uint,
        outputs: *mut rknn_output,
        extend: *const c_void,
    ) -> c_int;
    fn rknn_outputs_release(
        context: rknn_context,
        n_ouputs: c_uint,
        outputs: *mut rknn_output,
    ) -> c_int;
}

// ============================================================================
// Helpers
// ============================================================================

fn check(ret: c_int, msg: &str) -> Result<()> {
    if ret != RKNN_SUCC {
        Err(Error::Rknn(format!("{msg}: ret={ret}")))
    } else {
        Ok(())
    }
}

fn query_tensor(ctx: rknn_context, cmd: c_uint, index: c_uint) -> Result<rknn_tensor_attr> {
    let mut attr: rknn_tensor_attr = unsafe { std::mem::zeroed() };
    attr.index = index;
    check(
        unsafe {
            rknn_query(
                ctx,
                cmd,
                &mut attr as *mut _ as *mut c_void,
                std::mem::size_of_val(&attr) as c_uint,
            )
        },
        "rknn_query",
    )?;
    Ok(attr)
}

// ============================================================================
// DepthModel
// ============================================================================

/// Loaded RKNN model with input/output dimensions.
pub struct DepthModel {
    ctx: rknn_context,
    pub input_w: usize,
    pub input_h: usize,
    pub output_h: usize,
    pub output_w: usize,
}

/// NPU core selection for RK3588 (3 cores).
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum NpuCore {
    /// Let the runtime pick an idle core
    Auto,
    /// Pin to core 0
    Core0,
    /// Pin to core 1
    Core1,
    /// Pin to core 2
    Core2,
    /// Distribute across all three cores
    Core012,
}

impl NpuCore {
    fn mask(self) -> c_uint {
        match self {
            NpuCore::Auto => 0,
            NpuCore::Core0 => 1,
            NpuCore::Core1 => 2,
            NpuCore::Core2 => 4,
            NpuCore::Core012 => 7,
        }
    }
}

impl DepthModel {
    /// Load an RKNN model and query its I/O tensor shapes.
    pub fn load(model_path: &str, core: NpuCore) -> Result<Self> {
        let c_path = CString::new(model_path).map_err(|e| Error::Invalid(e.to_string()))?;

        let mut ctx: rknn_context = 0;
        check(
            unsafe {
                rknn_init(
                    &mut ctx as *mut _,
                    c_path.as_ptr() as *const c_void,
                    0,
                    0,
                    ptr::null(),
                )
            },
            "rknn_init",
        )?;

        if core != NpuCore::Auto {
            check(
                unsafe { rknn_set_core_mask(ctx, core.mask()) },
                "rknn_set_core_mask",
            )?;
            info!("NPU core: {:?}", core);
        }

        // Query I/O count
        let mut io_num: rknn_input_output_num = unsafe { std::mem::zeroed() };
        check(
            unsafe {
                rknn_query(
                    ctx,
                    RKNN_QUERY_IN_OUT_NUM,
                    &mut io_num as *mut _ as *mut c_void,
                    std::mem::size_of_val(&io_num) as c_uint,
                )
            },
            "rknn_query io_num",
        )?;
        info!("Model I/O: {} inputs, {} outputs", io_num.n_input, io_num.n_output);

        // Query input (NHWC: [N, H, W, C])
        let input_attr = query_tensor(ctx, RKNN_QUERY_INPUT_ATTR, 0)?;
        let input_h = input_attr.dims[1] as usize;
        let input_w = input_attr.dims[2] as usize;
        info!("Input: {}x{}", input_w, input_h);

        // Query output (NCHW: [N, C, H, W])
        let output_attr = query_tensor(ctx, RKNN_QUERY_OUTPUT_ATTR, 0)?;
        let output_h = output_attr.dims[2] as usize;
        let output_w = output_attr.dims[3] as usize;
        info!("Output: {}x{} (NCHW dims[2]x[3])", output_w, output_h);

        Ok(DepthModel { ctx, input_w, input_h, output_h, output_w })
    }

    /// Run inference. Returns flat f32 depth map of size output_w * output_h.
    pub fn infer(&self, rgb_data: &[u8]) -> Result<Vec<f32>> {
        let mut input: rknn_input = unsafe { std::mem::zeroed() };
        input.index = 0;
        input.buf = rgb_data.as_ptr() as *mut c_void;
        input.size = rgb_data.len() as c_uint;
        input.pass_through = 0;
        input.type_ = RKNN_TENSOR_UINT8 as c_uint;
        input.fmt = RKNN_TENSOR_NHWC as c_uint;

        check(unsafe { rknn_inputs_set(self.ctx, 1, &mut input) }, "rknn_inputs_set")?;
        check(unsafe { rknn_run(self.ctx, ptr::null()) }, "rknn_run")?;

        let mut output: rknn_output = rknn_output {
            want_float: 1,
            is_prealloc: 0,
            index: 0,
            buf: ptr::null_mut(),
            size: 0,
        };

        check(
            unsafe { rknn_outputs_get(self.ctx, 1, &mut output, ptr::null()) },
            "rknn_outputs_get",
        )?;

        let output_len = output.size as usize / std::mem::size_of::<f32>();
        let mut depth_raw = vec![0.0f32; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(output.buf as *const f32, depth_raw.as_mut_ptr(), output_len);
        }

        check(
            unsafe { rknn_outputs_release(self.ctx, 1, &mut output) },
            "rknn_outputs_release",
        )?;

        Ok(depth_raw)
    }
}

impl Drop for DepthModel {
    fn drop(&mut self) {
        unsafe { rknn_destroy(self.ctx); }
    }
}
