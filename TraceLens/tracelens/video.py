"""Video measurements: container metadata (ffprobe if present), sampled frames and a short
dense burst for temporal statistics. Uses OpenCV headless."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .imaging import scan_provenance, sha256_bytes


def _ffprobe(path: str) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        cp = subprocess.run([exe, "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                            capture_output=True, text=True, timeout=60)
        return json.loads(cp.stdout) if cp.returncode == 0 else {}
    except Exception:
        return {}


def _residual(gray: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gray - gaussian_filter(gray, 1.5)


def analyze_video(data: bytes, filename: str, n_frames: int = 6, burst: int = 24):
    import cv2
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    res: dict = {"media_type": "video", "filename": filename, "size_bytes": len(data), "sha256": sha256_bytes(data)}
    frames_rgb: list = []
    try:
        probe = _ffprobe(path)
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        res["container_tags"] = {**(fmt.get("tags") or {}),
                                 **{f"stream{ s.get('index')}.{k}": v for s in streams for k, v in (s.get("tags") or {}).items()
                                    if k in ("encoder", "handler_name", "vendor_id", "creation_time")}}
        res["format_name"] = fmt.get("format_name")
        res["codecs"] = [s.get("codec_name") for s in streams]
        res["has_audio"] = any(s.get("codec_type") == "audio" for s in streams) if probe else None
        res["ffprobe_available"] = bool(probe)
        res["provenance"] = scan_provenance(data[:3_000_000] + data[-3_000_000:] if len(data) > 6_000_000 else data)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open this video (unsupported codec).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        res.update({"fps": round(fps, 3), "frame_count": total, "width": w, "height": h,
                    "duration_seconds": round(total / fps, 2) if fps > 0 and total > 0 else None})

        def read_at(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            return f if ok else None

        if total > 0:
            idx = sorted({int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(min(n_frames, total))})
            for i in idx:
                f = read_at(i)
                if f is not None:
                    frames_rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            # dense burst around the middle for temporal statistics
            start = max(0, total // 2 - burst // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            grays = []
            for _ in range(min(burst, total)):
                ok, f = cap.read()
                if not ok:
                    break
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if max(g.shape) > 480:
                    s = 480 / max(g.shape)
                    g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)))
                grays.append(g)
            if len(grays) >= 4:
                diffs = [float(np.abs(a - b).mean()) for a, b in zip(grays, grays[1:])]
                means = np.array([g.mean() for g in grays])
                res_list = [_residual(g) for g in grays]
                corrs = []
                for a, b in zip(res_list, res_list[1:]):
                    if a.std() > 1e-6 and b.std() > 1e-6:
                        corrs.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
                res["temporal"] = {
                    "burst_frames": len(grays),
                    "mean_frame_diff": round(float(np.mean(diffs)), 3),
                    "flicker_cv": round(float(np.std(np.diff(means)) / (means.mean() + 1e-6)), 4),
                    "noise_temporal_corr": round(float(np.mean(corrs)), 3) if corrs else None,
                    "noise_std_cv_across_frames": round(float(np.std([r.std() for r in res_list]) /
                                                              (np.mean([r.std() for r in res_list]) + 1e-6)), 3),
                    "note": "Short burst from the middle of the clip only; not a whole-video analysis.",
                }
        cap.release()
    finally:
        Path(path).unlink(missing_ok=True)
    return res, frames_rgb
