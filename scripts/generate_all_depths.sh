#!/bin/bash
# Run on RK3588 (t6): generate depth outputs for all 5 models × 3 languages
# Usage: bash scripts/generate_all_depths.sh

set -e
cd "$(dirname "$0")/.."

IMG="assets/bus.jpg"
OUTDIR="assets/depths"
mkdir -p "$OUTDIR"

MODELS_RKNN=(
    "yolo26n-depth-float.rknn"
    "yolo26s-depth-float.rknn"
    "yolo26m-depth-float.rknn"
    "yolo26l-depth-float.rknn"
    "yolo26x-depth-float.rknn"
)

SIZES=("n" "s" "m" "l" "x")

# ========== Python RKNN ==========
echo "=== Python RKNN ==="
for i in "${!MODELS_RKNN[@]}"; do
    m="${MODELS_RKNN[$i]}"
    s="${SIZES[$i]}"
    if [ -f "$m" ]; then
        echo "  $s ..."
        python3 python/rknn/inference_rknn.py \
            --model "$m" --image "$IMG" \
            --save-heat "$OUTDIR/py_${s}_heat.png" \
            --save-depth "$OUTDIR/py_${s}_depth.npy" \
            --save "$OUTDIR/py_${s}_overlay.png" 2>/dev/null
    fi
done

# ========== Rust ==========
echo "=== Rust RKNN ==="
for i in "${!MODELS_RKNN[@]}"; do
    m="${MODELS_RKNN[$i]}"
    s="${SIZES[$i]}"
    if [ -f "$m" ]; then
        echo "  $s ..."
        ./rust/target/release/yolo26depth-rknn \
            --model "$m" --image "$IMG" \
            --output "$OUTDIR/rs_${s}_heat.png" \
            --save-depth "$OUTDIR/rs_${s}_depth.bin" 2>/dev/null
    fi
done

# ========== C++ ==========
echo "=== C++ RKNN ==="
for i in "${!MODELS_RKNN[@]}"; do
    m="${MODELS_RKNN[$i]}"
    s="${SIZES[$i]}"
    if [ -f "$m" ]; then
        echo "  $s ..."
        ./cpp/src/build/yolo26depth-cpp \
            --model "$m" --image "$IMG" \
            --save-heat "$OUTDIR/cpp_${s}_heat.png" \
            --save-depth "$OUTDIR/cpp_${s}_depth.bin" \
            --save "$OUTDIR/cpp_${s}_overlay.png" 2>/dev/null
    fi
done

echo "=== All done ==="
ls -la "$OUTDIR/"
