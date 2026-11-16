import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from tracelens.ai import apply_guardrails, normalise, _extract_json
from tracelens.imaging import analyze_image, jpeg_quant_info
from tracelens.signals import image_signals


def _jpeg(q=80):
    arr = (np.random.default_rng(0).normal(128, 40, (128, 128, 3))).clip(0, 255).astype("uint8")
    b = io.BytesIO(); Image.fromarray(arr).save(b, "JPEG", quality=q); return b.getvalue()


def test_quality_estimate():
    for q in (30, 75, 95):
        assert jpeg_quant_info(Image.open(io.BytesIO(_jpeg(q))))["estimated_quality"] == q


def test_missing_exif_is_neutral():
    res, _, _ = analyze_image(_jpeg(), "a.jpg")
    sig = {s["id"]: s for s in image_signals(res)}
    assert sig["no_exif"]["direction"] == "neutral" and sig["no_exif"]["weight"] == 0


def test_ai_metadata_overrides_model():
    v = apply_guardrails(normalise({"category": "camera_capture", "is_photographic": True, "confidence": 95}),
                         [{"id": "declared_ai", "direction": "generated", "weight": 10}])
    assert v["category"] == "ai_generated"


def test_camera_claim_without_metadata_is_capped():
    v = apply_guardrails(normalise({"category": "camera_capture", "is_photographic": True, "confidence": 98}), [])
    assert v["confidence"] <= 70


def test_json_extraction():
    assert _extract_json('<think>x</think>```json\n{"a": 1}\n```') == {"a": 1}
