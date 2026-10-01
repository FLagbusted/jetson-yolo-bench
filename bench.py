"""Export each model to TensorRT (FP32, FP16, INT8) and measure latency + mAP50-95 against PyTorch."""
import argparse, csv, json, time, platform
from pathlib import Path
import numpy as np
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["yolo11n"])
p.add_argument("--data", default="coco128.yaml")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--runs", type=int, default=200)
p.add_argument("--variants", nargs="+", default=["pytorch-fp32", "tensorrt-fp32", "tensorrt-fp16", "tensorrt-int8"])
a = p.parse_args()
out = Path("results"); out.mkdir(exist_ok=True)
rows = []

def latency(model, n):
    img = np.random.randint(0, 255, (a.imgsz, a.imgsz, 3), dtype=np.uint8)
    for _ in range(20):
        model.predict(img, imgsz=a.imgsz, verbose=False)
    t = []
    for _ in range(n):
        s = time.perf_counter(); model.predict(img, imgsz=a.imgsz, verbose=False); t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return float(np.median(t)), float(np.percentile(t, 95))

for m in a.models:
    variants = [("pytorch-fp32", lambda: YOLO(f"{m}.pt"))]
    for prec in ["fp32", "fp16", "int8"]:
        def mk(prec=prec):
            kw = dict(format="engine", imgsz=a.imgsz, device=0, workspace=4)
            if prec == "fp16": kw["half"] = True
            if prec == "int8": kw.update(int8=True, data=a.data)
            path = YOLO(f"{m}.pt").export(**kw)
            target = Path(f"{m}-{prec}.engine"); Path(path).rename(target)
            return YOLO(str(target), task="detect")
        variants.append((f"tensorrt-{prec}", mk))
    for name, build in variants:
        if name not in a.variants: continue
        print(f"== {m} {name}", flush=True)
        try:
            t0 = time.time(); model = build(); build_s = time.time() - t0
            p50, p95 = latency(model, a.runs)
            mp = model.val(data=a.data, imgsz=a.imgsz, batch=1, device=0, verbose=False, plots=False).box.map
            rows.append(dict(model=m, variant=name, p50_ms=round(p50, 2), p95_ms=round(p95, 2), fps=round(1000 / p50, 1), map50_95=round(float(mp), 4), build_s=round(build_s, 1)))
        except Exception as e:
            rows.append(dict(model=m, variant=name, error=str(e)[:200]))
        print(rows[-1], flush=True)
        with open(out / "results.json", "w") as f: json.dump(rows, f, indent=2)

keys = ["model", "variant", "p50_ms", "p95_ms", "fps", "map50_95", "build_s", "error"]
with open(out / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
print("done ->", out / "results.csv")
