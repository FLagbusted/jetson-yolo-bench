#!/usr/bin/env bash
# YOLO11 on Jetson AGX Orin: PyTorch vs TensorRT FP32/FP16/INT8, measured on the device.
# Run on the Orin (JetPack 6.x). Takes ~30-60 min the first time (engine builds are slow).
set -euo pipefail
cd "$(dirname "$0")"
MODELS="${MODELS:-yolo11n yolo11s yolo11m}"
DATA="${DATA:-coco128.yaml}"      # small COCO subset, auto-downloaded by ultralytics
IMGSZ="${IMGSZ:-640}"
mkdir -p results
# TensorRT export goes through ONNX; install the exporter deps up front (first run failed without them)
python3 -c "import onnx, onnxslim" 2>/dev/null || pip3 install "onnx>=1.12" onnxslim
python3 -c "import onnx, onnxslim; print('onnx', onnx.__version__)"
VARIANTS="${VARIANTS:-pytorch-fp32 tensorrt-fp32 tensorrt-fp16 tensorrt-int8}"
{ echo "# Device"; cat /etc/nv_tegra_release 2>/dev/null || true; uname -a;
  sudo nvpmodel -q 2>/dev/null || true; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())";
  python3 -c "import tensorrt;print('tensorrt',tensorrt.__version__)" 2>/dev/null || echo "tensorrt python not found"; } > results/device.txt 2>&1
sudo nvpmodel -m 0 2>/dev/null || true   # MAXN power mode
sudo jetson_clocks 2>/dev/null || true
python3 bench.py --models $MODELS --data "$DATA" --imgsz "$IMGSZ" --variants $VARIANTS
tar czf ../jetson-bench-results.tgz results
echo "Upload this file back to Claude: $(cd .. && pwd)/jetson-bench-results.tgz"
