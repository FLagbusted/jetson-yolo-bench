# jetson-yolo-bench

YOLO11 on an NVIDIA Jetson AGX Orin 64GB: PyTorch vs TensorRT FP32 / FP16 / INT8, measured on the device.

**Device:** Jetson AGX Orin 64GB Developer Kit · JetPack 6 (L4T R36.5) · MAXN power mode with `jetson_clocks` · PyTorch 2.8 (CUDA) · TensorRT 10.3 · Ultralytics YOLO11 · 640×640 input, batch 1.

## Results

| Model | Runtime | p50 latency (ms) | p95 (ms) | FPS (from p50) | Speed-up vs PyTorch | mAP50-95 (COCO128) | Δ mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | PyTorch FP32 | 22.48 | 24.57 | 44.5 | 1.0× | 0.507 | — |
| YOLO11n | TensorRT FP32 | 6.92 | 7.85 | 144.4 | 3.2× | 0.506 | −0.002 |
| YOLO11n | TensorRT FP16 | 5.59 | 6.45 | 178.9 | 4.0× | 0.505 | −0.003 |
| YOLO11n | TensorRT INT8 | 5.34 | 8.34 | 187.4 | 4.2× | 0.469 | −0.038 |
| YOLO11s | PyTorch FP32 | 23.31 | 25.62 | 42.9 | 1.0× | 0.574 | — |
| YOLO11s | TensorRT FP32 | 9.74 | 10.81 | 102.7 | 2.4× | 0.575 | +0.001 |
| YOLO11s | TensorRT FP16 | 8.28 | 9.65 | 120.8 | 2.8× | 0.576 | +0.001 |
| YOLO11s | TensorRT INT8 | 5.76 | 6.77 | 173.5 | 4.0× | 0.570 | −0.004 |
| YOLO11m | PyTorch FP32 | 28.09 | 33.78 | 35.6 | 1.0× | 0.622 | — |
| YOLO11m | TensorRT FP32 | 21.05 | 23.11 | 47.5 | 1.3× | 0.627 | +0.005 |
| YOLO11m | TensorRT FP16 | 9.08 | 9.21 | 110.1 | 3.1× | 0.627 | +0.005 |
| YOLO11m | TensorRT INT8 | 7.48 | 7.60 | 133.7 | 3.8× | 0.625 | +0.004 |

Raw data: `results/results.csv`, `results/device.txt`.

## What the numbers say

- **FP16 is the safe default.** It gave a 2.8–4.0× speed-up over PyTorch with no measurable accuracy change on any model.
- **INT8 helps most on the bigger models.** YOLO11m went from 9.1 ms (FP16) to 7.5 ms. On YOLO11n, INT8 bought only 0.25 ms over FP16 but cost 0.038 mAP and had a worse p95, so FP16 is the better choice for the nano model.
- **In PyTorch, model size barely changes latency** (22.5 ms for n vs 28.1 ms for m). Most of the time is framework and pre/post-processing overhead, which TensorRT removes.
- **YOLO11m at FP16/INT8 runs above 100 FPS**, 2.5–3× faster than the nano model in PyTorch, at a much higher accuracy.

## How it was measured

- Latency: 20 warm-up runs, then 200 timed `model.predict()` calls on one 640×640 image. The timing is end to end through Ultralytics, so it includes pre-processing, inference and NMS, not just the engine.
- Accuracy: `model.val()` on COCO128 at batch 1. INT8 engines were calibrated on the same dataset.
- PyTorch rows come from run 1 and TensorRT rows from run 2 on the same device, power mode and software stack (run 1's TensorRT export failed because the `onnx` package was missing; the script now installs it).

## Limits

- COCO128 is a 128-image slice of the COCO **training** set, so the mAP values are inflated. Use them to compare runtimes against each other, not as real accuracy. INT8 calibration on the same images also flatters INT8.
- Single image size and batch size 1. Real pipelines with camera decode or DeepStream batching will behave differently.
- Engine build times (105–420 s) are one-off costs on the device.

## Reproduce

    pip3 install ultralytics          # JetPack's CUDA-enabled torch/torchvision must already be installed
    ./bench.sh                        # MODELS, DATA, IMGSZ, VARIANTS env vars are optional

## License

MIT
