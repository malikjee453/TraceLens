"""Measured evidence for still images. Everything here is local (no network)."""
from __future__ import annotations

import hashlib
import io
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageChops, ImageEnhance

# ---------------------------------------------------------------- constants
GENERATOR_KEYWORDS = [
    "midjourney", "dall-e", "dalle", "openai", "stable diffusion", "stablediffusion",
    "sdxl", "comfyui", "automatic1111", "invokeai", "firefly", "adobe firefly",
    "leonardo.ai", "ideogram", "imagen", "gemini", "flux", "nano banana",
    "bing image creator", "dreamstudio", "novelai", "playground ai", "canva ai",
    "synthid", "generative fill", "generative ai",
]
# IPTC digital source type values (https://cv.iptc.org/newscodes/digitalsourcetype)
IPTC_AI = ["trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia",
           "algorithmicmedia", "compositesynthetic"]
IPTC_CAMERA = ["digitalcapture", "computationalcapture", "negativefilm", "positivefilm"]
GENERATOR_SIZES = {(512, 512), (768, 768), (1024, 1024), (1024, 1792), (1792, 1024),
                   (1344, 768), (768, 1344), (1536, 1024), (1024, 1536), (832, 1216),
                   (1216, 832), (896, 1152), (1152, 896), (2048, 2048), (1536, 1536)}
# Standard IJG luminance quantisation table (quality 50 baseline)
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")[:300]
    if isinstance(v, (tuple, list)):
        return [_clean(x) for x in v]
    if hasattr(v, "numerator") and hasattr(v, "denominator"):  # IFDRational
        try:
            return float(v)
        except Exception:
            return str(v)
    return v if isinstance(v, (int, float, str)) else str(v)


# ---------------------------------------------------------------- metadata
def read_exif(im: Image.Image) -> dict:
    """Reads IFD0 + Exif sub-IFD + GPS sub-IFD. (v1.0 only read IFD0, so it missed
    DateTimeOriginal, ExposureTime, ISO, LensModel, etc.)"""
    out: dict = {}
    try:
        raw = im.getexif()
    except Exception as e:  # pragma: no cover
        return {"_exif_read_error": str(e)}
    for k, v in raw.items():
        out[ExifTags.TAGS.get(k, str(k))] = _clean(v)
    for ifd_id, label in ((0x8769, "Exif"), (0x8825, "GPS")):
        try:
            sub = raw.get_ifd(ifd_id)
        except Exception:
            sub = {}
        tags = ExifTags.GPSTAGS if label == "GPS" else ExifTags.TAGS
        for k, v in sub.items():
            name = tags.get(k, str(k))
            out[("GPS." if label == "GPS" else "") + name] = _clean(v)
    return out


def scan_provenance(data: bytes) -> dict:
    """Byte-level scan for C2PA/JUMBF, XMP, IPTC digital-source-type, AI-tool strings and
    PNG text chunks. NOTE: this detects *presence*; it does not verify signatures."""
    low = data.lower()
    head_tail = low[:2_000_000] + low[-2_000_000:] if len(low) > 4_000_000 else low
    res: dict = {
        "c2pa_manifest_present": (b"c2pa" in head_tail) or (b"jumb" in head_tail and b"c2pa" in head_tail),
        "jumbf_present": b"jumb" in head_tail,
        "xmp_present": b"<x:xmpmeta" in head_tail or b"http://ns.adobe.com/xap" in head_tail,
        "iptc_ai_source_type": [s for s in IPTC_AI if s.encode() in head_tail],
        "iptc_camera_source_type": [s for s in IPTC_CAMERA if s.encode() in head_tail],
        "generator_strings": sorted({k for k in GENERATOR_KEYWORDS if k.encode() in head_tail}),
        "sd_parameters_chunk": bool(re.search(rb"(steps:\s*\d+.*sampler:|negative prompt:)", head_tail, re.S)),
        "signature_verified": False,
        "note": "Presence only. Cryptographic C2PA verification is NOT performed in this build.",
    }
    # who signed it? crude, but useful when present
    m = re.search(rb"claim_generator[^a-z0-9]{0,6}([a-z0-9 ._/\-]{3,60})", head_tail)
    if m:
        res["claim_generator_hint"] = m.group(1).decode("latin-1").strip()
    return res


def jpeg_quant_info(im: Image.Image) -> dict:
    q = getattr(im, "quantization", None)
    if not q or im.format != "JPEG":
        return {}
    luma = np.array(q.get(0, []), dtype=float)
    info = {"tables": len(q)}
    if luma.size == 64:
        best = None
        for quality in range(1, 101):  # Pillow returns tables in natural order; libjpeg uses integer maths
            s = 5000 // quality if quality < 50 else 200 - 2 * quality
            cand = np.clip((_STD_LUMA * s + 50) // 100, 1, 255)
            err = float(np.abs(cand - luma).mean())
            if best is None or err < best[1]:
                best = (quality, err)
        info["estimated_quality"] = best[0]
        info["matches_standard_ijg_table"] = best[1] < 0.25
        info["note"] = ("Standard IJG tables are typical of software/web re-saves (Pillow, libjpeg, "
                        "browsers, editors). Cameras and phones often use custom tables.")
    return info




# ---------------------------------------------------------------- pixel analyses
def _gray(im: Image.Image, max_side: int = 1024) -> np.ndarray:
    g = im.convert("L")
    if max(g.size) > max_side:
        s = max_side / max(g.size)
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))), Image.LANCZOS)
    return np.asarray(g, dtype=np.float32)


def _block_stats(a: np.ndarray, b: int = 16) -> np.ndarray:
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    blocks = a[:h, :w].reshape(h // b, b, w // b, b).swapaxes(1, 2).reshape(-1, b * b)
    return blocks


def ela_analysis(im: Image.Image, quality: int = 90, scale: int = 10):
    """Returns (stats, ela_image). Only meaningful for JPEG sources."""
    rgb = im.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re_img = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(rgb, re_img)
    arr = np.asarray(diff, dtype=np.float32).mean(axis=2)
    blk = _block_stats(arr).mean(axis=1)
    stats = {
        "comparison_quality": quality, "display_scale": scale,
        "mean_error": round(float(arr.mean()), 4),
        "p99_error": round(float(np.percentile(arr, 99)), 3),
        "block_error_cv": round(float(blk.std() / (blk.mean() + 1e-6)), 3),
        "note": "ELA is a visual aid. Uneven regions have many benign causes (sharp edges, text, "
                "saturated colours, prior recompression).",
    }
    return stats, ImageEnhance.Brightness(diff).enhance(scale)


def noise_analysis(im: Image.Image) -> dict:
    """High-pass residual statistics. Real sensors leave spatially varying, brightness-dependent
    noise; synthetic/flat-shaded images are usually either almost noise-free or have
    unnaturally uniform noise."""
    from scipy.ndimage import gaussian_filter  # scipy is a light dependency
    g = _gray(im)
    resid = g - gaussian_filter(g, 1.5)
    blocks = _block_stats(resid, 16)
    bstd = blocks.std(axis=1)
    lum = _block_stats(g, 16).mean(axis=1)
    flat = float((_block_stats(g, 8).std(axis=1) < 1.0).mean())
    corr = float(np.corrcoef(lum, bstd)[0, 1]) if bstd.std() > 1e-6 and lum.std() > 1e-6 else 0.0
    return {
        "residual_std": round(float(resid.std()), 3),
        "block_noise_median": round(float(np.median(bstd)), 3),
        "block_noise_cv": round(float(bstd.std() / (bstd.mean() + 1e-6)), 3),
        "noise_vs_brightness_corr": round(corr, 3),
        "flat_block_fraction": round(flat, 3),
    }


def frequency_analysis(im: Image.Image) -> dict:
    """Radial power-spectrum features. Some generators/upsamplers leave periodic peaks.
    Weak signal: resizing and JPEG compression also create peaks."""
    g = _gray(im, 512)
    n = min(g.shape)
    g = g[:n, :n] * np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.fft.fftshift(np.fft.fft2(g))
    p = np.log1p(np.abs(f) ** 2)
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    radial = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    radial = radial[: n // 2]
    hf = float(radial[int(len(radial) * 0.66):].mean() / (radial[: int(len(radial) * 0.33)].mean() + 1e-6))
    ring = (r > n * 0.25) & (r < n * 0.5)
    resid = p - np.take(radial, np.clip(r, 0, len(radial) - 1))
    vals = resid[ring]
    peak = float((vals.max() - np.median(vals)) / (vals.std() + 1e-6))
    return {"high_to_low_freq_ratio": round(hf, 3), "spectral_peak_score": round(peak, 2),
            "note": "Weak signal; high peak scores can come from resizing or compression."}


def stylization_analysis(im: Image.Image) -> dict:
    """Distinguishes flat/graphic imagery (cartoons, vectors, UI, ads) from photographic detail."""
    small = im.convert("RGB")
    small.thumbnail((400, 400))
    a = np.asarray(small)
    q = (a // 8).reshape(-1, 3)
    uniq = len(np.unique(q[:, 0].astype(np.int32) * 1024 + q[:, 1] * 32 + q[:, 2]))
    top = np.unique(q[:, 0].astype(np.int32) * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)[1]
    top_share = float(np.sort(top)[::-1][:16].sum() / q.shape[0])
    g = _gray(im, 512)
    gx, gy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    strong = float(((gx > 60).mean() + (gy > 60).mean()) / 2)
    return {"distinct_colours_32level": int(uniq), "top16_colour_share": round(top_share, 3),
            "strong_edge_fraction": round(strong, 4)}


def dimension_hints(w: int, h: int) -> dict:
    return {"is_common_generator_size": (w, h) in GENERATOR_SIZES,
            "both_multiples_of_64": w % 64 == 0 and h % 64 == 0,
            "megapixels": round(w * h / 1e6, 2)}


# ---------------------------------------------------------------- orchestrator
def analyze_image(data: bytes, filename: str) -> tuple:
    im = Image.open(io.BytesIO(data))
    im.load()
    fmt = im.format
    res: dict = {
        "media_type": "image", "filename": filename, "size_bytes": len(data),
        "sha256": sha256_bytes(data), "format_detected": fmt,
        "width": im.width, "height": im.height, "mode": im.mode,
        "has_alpha": "A" in im.mode,
    }
    exif = read_exif(im)
    res["exif"] = exif
    res["png_text_chunks"] = {k: str(v)[:400] for k, v in (getattr(im, "text", {}) or {}).items()}
    res["provenance"] = scan_provenance(data)
    res["dimension_hints"] = dimension_hints(im.width, im.height)
    res["jpeg"] = jpeg_quant_info(im)
    res["noise"] = noise_analysis(im)
    res["frequency"] = frequency_analysis(im)
    res["stylization"] = stylization_analysis(im)
    ela_img = None
    if fmt == "JPEG":
        res["ela"], ela_img = ela_analysis(im)
    else:
        res["ela"] = {"note": "ELA is only produced for JPEG sources."}
    return res, im, ela_img
