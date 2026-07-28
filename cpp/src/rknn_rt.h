#ifndef RKNN_RT_H_
#define RKNN_RT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Return codes */
#define RKNN_SUCC                0

/* Query commands */
#define RKNN_QUERY_IN_OUT_NUM    0
#define RKNN_QUERY_INPUT_ATTR    1
#define RKNN_QUERY_OUTPUT_ATTR   2

/* Tensor types / formats */
#define RKNN_TENSOR_UINT8        3
#define RKNN_TENSOR_NHWC         1
#define RKNN_TENSOR_NCHW         2

/* NPU core mask for RK3588 (3 cores) */
typedef enum {
    NPU_CORE_AUTO = 0,
    NPU_CORE_0    = 1,
    NPU_CORE_1    = 2,
    NPU_CORE_2    = 4,
    NPU_CORE_0_1_2 = 7,
} rknn_core_mask;

typedef uint64_t rknn_context;

/* Tensor attribute (matches rknn-toolkit-lite2 runtime) */
#define RKNN_MAX_DIMS     16
#define RKNN_MAX_NAME_LEN 256

typedef struct {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    char     name[RKNN_MAX_NAME_LEN];
    uint32_t n_elems;
    uint32_t size;
    uint32_t fmt;       /* RKNN_TENSOR_NHWC / NCHW */
    uint32_t type_;     /* RKNN_TENSOR_UINT8 / FLOAT32 etc */
    uint32_t qnt_type;
    int8_t   fl;        /* quantization scale factor? */
    int32_t  zp;        /* zero point */
    float    scale;     /* quantization scale */
    uint32_t w_stride;
    uint32_t size_with_stride;
    uint8_t  pass_through;
    uint32_t h_stride;
} rknn_tensor_attr;

/* I/O count */
typedef struct {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

/* Model input (NHWC: N,H,W,C layout expected) */
typedef struct {
    uint32_t index;
    void*    buf;
    uint32_t size;
    uint8_t  pass_through;
    uint32_t type_;   /* RKNN_TENSOR_UINT8 */
    uint32_t fmt;     /* RKNN_TENSOR_NHWC */
} rknn_input;

/* Model output */
typedef struct {
    uint8_t  want_float;
    uint8_t  is_prealloc;
    uint32_t index;
    void*    buf;
    uint32_t size;
} rknn_output;

/* Core functions (as used by Rust binding on rk3588) */
int  rknn_init(rknn_context* context, const char* model_path, uint32_t size, uint32_t flag, const void* extend);
int  rknn_destroy(rknn_context context);
int  rknn_set_core_mask(rknn_context context, uint32_t core_mask);
int  rknn_query(rknn_context context, uint32_t cmd, void* info, uint32_t size);
int  rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input* inputs);
int  rknn_run(rknn_context context, const void* extend);
int  rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output* outputs, const void* extend);
void rknn_outputs_release(rknn_context context, uint32_t n_ouputs, rknn_output* outputs);

#ifdef __cplusplus
}
#endif

#endif /* RKNN_RT_H_ */
