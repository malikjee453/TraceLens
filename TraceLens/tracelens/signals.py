"""Turns raw measurements into a small list of weighted, human-readable signals.
These are *weak, documented heuristics*. They are shown to the examiner and fed to the LLM
as context; they are never presented as a verdict on their own."""
from __future__ import annotations

EDITORS = ["photoshop", "gimp", "lightroom", "canva", "affinity", "pixelmator", "snapseed",
           "paint.net", "capcut", "premiere", "final cut", "davinci", "lavf", "lavc", "ffmpeg", "handbrake"]
CAMERA_MAKERS = ["canon", "nikon", "sony", "fujifilm", "fuji", "panasonic", "olympus", "leica", "pentax",
                 "apple", "samsung", "google", "huawei", "xiaomi", "oppo", "vivo", "oneplus", "motorola",
                 "dji", "gopro", "insta360", "hasselblad", "ricoh", "sigma"]


def _s(sid, label, direction, weight, why):
    return {"id": sid, "label": label, "direction": direction, "weight": weight, "why": why}


# Pixel-statistics camera signals: weaker than metadata, so the guardrails treat them separately.
PIXEL_CAMERA_IDS = {"natural_noise", "shot_noise", "cfa"}


def image_signals(r: dict) -> list[dict]:
    out: list[dict] = []
    exif, prov = r.get("exif", {}), r.get("provenance", {})

    # ---- declared provenance (strongest class)
    ai_flags = list(prov.get("iptc_ai_source_type", [])) + list(prov.get("generator_strings", []))
    if prov.get("sd_parameters_chunk"):
        ai_flags.append("Stable-Diffusion-style parameters")
    if r.get("png_text_chunks"):
        for k, v in r["png_text_chunks"].items():
            if any(g in (k + " " + v).lower() for g in ("prompt", "workflow", "sampler", "diffusion")):
                ai_flags.append(f"PNG text chunk '{k}'")
    if ai_flags:
        out.append(_s("declared_ai", "Metadata declares AI generation/tooling", "generated", 10,
                      "Found: " + ", ".join(sorted(set(map(str, ai_flags)))) +
                      ". Metadata can be forged or copied, but this is the strongest class of evidence."))
    if prov.get("c2pa_manifest_present"):
        hint = prov.get("claim_generator_hint", "")
        out.append(_s("c2pa", "C2PA / Content Credentials data present", "neutral", 2,
                      "A provenance manifest exists" + (f" (claim generator hint: {hint})" if hint else "") +
                      ". Signature NOT verified here — verify with a C2PA tool before relying on it."))
    if prov.get("iptc_camera_source_type"):
        out.append(_s("iptc_camera", "IPTC source type says camera capture", "camera", 3,
                      "Declared: " + ", ".join(prov["iptc_camera_source_type"]) + " (self-declared, forgeable)."))

    # ---- camera EXIF
    make, model = str(exif.get("Make", "")).strip(), str(exif.get("Model", "")).strip()
    if make or model:
        known = any(m in (make + " " + model).lower() for m in CAMERA_MAKERS)
        out.append(_s("camera_make", "Camera make/model in EXIF", "camera", 3 if known else 1.5,
                      f"Make='{make}', Model='{model}'. EXIF can be edited or forged."))
    shoot = [k for k in ("ExposureTime", "FNumber", "ISOSpeedRatings", "PhotographicSensitivity",
                         "FocalLength", "LensModel", "ExposureProgram", "MeteringMode") if k in exif]
    if len(shoot) >= 3:
        out.append(_s("shooting_params", "Detailed shooting parameters", "camera", 2,
                      "Present: " + ", ".join(shoot)))
    if "MakerNote" in exif:
        out.append(_s("makernote", "Manufacturer MakerNote present", "camera", 1.5,
                      "Proprietary camera block; rarely preserved by generators or editors."))
    if any(k.startswith("GPS.") for k in exif):
        out.append(_s("gps", "GPS data present", "camera", 1, "Usually from a phone/camera; also forgeable."))
    if r.get("exif_tag_count", 0) >= 25:
        out.append(_s("exif_rich", f"Rich EXIF block ({r['exif_tag_count']} tags)", "camera", 1,
                      "Camera/phone files carry many tags; editors and generators usually keep few."))
    if r.get("exif_thumbnail_present"):
        out.append(_s("exif_thumb", "Embedded EXIF thumbnail", "camera", 0.7,
                      "Cameras embed a thumbnail; generators rarely do (some editors keep it)."))
    sw = str(exif.get("Software", "")).strip()
    if sw:
        low = sw.lower()
        if any(g in low for g in ("firefly", "midjourney", "dall", "diffusion", "gemini", "openai")):
            out.append(_s("software_ai", "Software tag names an AI tool", "generated", 8, f"Software='{sw}'"))
        elif any(e in low for e in EDITORS):
            out.append(_s("software_editor", "Editing software tag", "edited", 1.5, f"Software='{sw}'"))
        else:
            out.append(_s("software_other", "Software tag present", "neutral", 0, f"Software='{sw}'"))
    if exif.get("DateTimeOriginal") and exif.get("DateTime") and str(exif["DateTimeOriginal"]) != str(exif["DateTime"]):
        out.append(_s("date_mismatch", "EXIF capture time differs from last-modified time", "edited", 1,
                      f"DateTimeOriginal={exif['DateTimeOriginal']} vs DateTime={exif['DateTime']} "
                      "(normal if the file was re-saved)."))
    if not exif:
        out.append(_s("no_exif", "No EXIF metadata", "neutral", 0,
                      "Missing EXIF is NOT evidence of fakeness: social media, messengers and screenshots strip it."))

    # ---- dimensions
    d = r.get("dimension_hints", {})
    if d.get("is_common_generator_size"):
        out.append(_s("gen_size", "Size matches a common AI-generator output", "generated", 1.5,
                      f"{r['width']}×{r['height']} is a typical generator size (also common for crops/resizes)."))
    elif d.get("both_multiples_of_64"):
        out.append(_s("mult64", "Both sides are multiples of 64", "generated", 0.8,
                      "Diffusion models work on 64-px grids; weak hint only."))

    # ---- JPEG structure
    j = r.get("jpeg", {})
    if j.get("matches_standard_ijg_table"):
        out.append(_s("jpeg_std", f"Standard JPEG tables (≈Q{j.get('estimated_quality')})", "neutral", 0,
                      "Typical of software/web re-saves; cameras often use custom tables. Not decisive."))
    elif j:
        out.append(_s("jpeg_custom", "Non-standard JPEG tables", "neutral", 0,
                      "Custom tables occur in cameras/phones and in editors such as Photoshop. Not decisive."))

    # ---- pixel statistics
    n, st, fr = r.get("noise", {}), r.get("stylization", {}), r.get("frequency", {})
    if n and st:
        if n.get("flat_block_fraction", 0) > 0.4 or st.get("top16_colour_share", 0) > 0.45:
            out.append(_s("flat_graphic", "Large flat-colour areas / small palette", "graphic", 2,
                          f"flat blocks={n['flat_block_fraction']}, top-16 colour share={st['top16_colour_share']}. "
                          "Typical of cartoons, vectors, UI and ads."))
        if n.get("block_noise_median", 0) < 1.2 and n.get("flat_block_fraction", 0) > 0.15:
            out.append(_s("too_clean", "Almost no sensor-like noise", "generated", 1,
                          "Very clean pixels; also true of heavily denoised or rendered images."))
        elif (1.5 <= n.get("block_noise_median", 0) <= 10 and n.get("flat_block_fraction", 1) < 0.15
              and 0.2 <= n.get("block_noise_cv", 0) <= 1.3):
            out.append(_s("natural_noise", "Spatially varying fine-grain noise", "camera", 0.8,
                          "Weak: textured scenes and some generators also look like this."))
    sn = r.get("sensor", {})
    corr = sn.get("noise_mean_var_corr")
    if corr is not None and corr > 0.3 and sn.get("smooth_blocks", 0) >= 25:
        out.append(_s("shot_noise", "Noise grows with brightness (sensor shot-noise pattern)", "camera", 1.2,
                      f"corr={corr} over {sn['smooth_blocks']} smooth blocks. Physically typical of sensors; "
                      "thresholds are unvalidated and heavy denoising removes it."))
        if sn.get("cfa_peak_ratio", 0) > 3.5:
            out.append(_s("cfa", "Demosaicing (colour-filter) periodicity", "camera", 0.6,
                          f"peak ratio={sn['cfa_peak_ratio']}. Only counted because shot-noise also agrees; "
                          "dotted/halftone graphics can fake this alone."))
    if fr.get("spectral_peak_score", 0) > 9:
        out.append(_s("spectral_peaks", "Periodic peaks in frequency spectrum", "generated", 1,
                      f"peak score {fr['spectral_peak_score']}; resizing/compression can cause this too."))
    ela = r.get("ela", {})
    if ela.get("block_error_cv", 0) > 1.6:
        out.append(_s("ela_uneven", "Uneven recompression error (ELA)", "edited", 1,
                      f"block CV {ela['block_error_cv']}. Sharp edges/text also raise this."))
    if r.get("has_alpha"):
        out.append(_s("alpha", "Has an alpha/transparency channel", "graphic", 1.5,
                      "Cameras never write alpha; common for graphics and cut-outs."))
    return out


def video_signals(v: dict) -> list[dict]:
    out: list[dict] = []
    tags = {k.lower(): str(val) for k, val in (v.get("container_tags") or {}).items()}
    blob = " ".join(tags.values()).lower()
    prov = v.get("provenance", {})
    ai_flags = prov.get("iptc_ai_source_type", []) + prov.get("generator_strings", [])
    if ai_flags:
        out.append(_s("declared_ai", "Metadata declares AI generation/tooling", "generated", 10,
                      "Found: " + ", ".join(sorted(set(ai_flags)))))
    if prov.get("c2pa_manifest_present"):
        out.append(_s("c2pa", "C2PA / Content Credentials data present", "neutral", 2,
                      "Signature NOT verified here."))
    dev = [k for k in tags if any(x in k for x in ("make", "model", "software", "com.apple"))]
    if any(m in blob for m in CAMERA_MAKERS) and dev:
        out.append(_s("device_tags", "Device make/model tags in container", "camera", 3,
                      "Tags: " + ", ".join(dev[:5]) + " (forgeable)."))
    if any(e in blob for e in ("lavf", "lavc", "ffmpeg", "handbrake", "capcut", "premiere", "davinci")):
        out.append(_s("reencoded", "Re-encoded by an editing/transcoding tool", "edited", 1,
                      "Encoder string indicates processing after capture (also true of nearly all web/social videos)."))
    if v.get("has_audio") is False:
        out.append(_s("no_audio", "No audio track", "generated", 0.7, "Many AI video tools output silent clips; so do many screen/CCTV clips."))
    if not tags:
        out.append(_s("no_tags", "No container metadata", "neutral", 0, "Common after social-media transcoding."))
    t = v.get("temporal", {})
    if t:
        if t.get("noise_temporal_corr", 0) > 0.35:
            out.append(_s("baked_noise", "Noise pattern repeats frame-to-frame", "generated", 1.5,
                          f"corr={t['noise_temporal_corr']}. Real sensor noise is independent per frame; "
                          "heavy compression or static scenes can also raise this."))
        if t.get("flicker_cv", 0) > 0.02:
            out.append(_s("flicker", "Brightness flicker between consecutive frames", "generated", 1,
                          f"CV={t['flicker_cv']}. Artificial lighting or auto-exposure also flickers."))
        if t.get("mean_frame_diff", 99) < 0.6:
            out.append(_s("static", "Near-static clip", "neutral", 0, "Too little motion for temporal tests to say much."))
    return out


def leaning(signals: list[dict]) -> dict:
    s = {"camera": 0.0, "generated": 0.0, "graphic": 0.0, "edited": 0.0}
    for x in signals:
        if x["direction"] in s:
            s[x["direction"]] += x["weight"]
    top = max(s, key=s.get)
    ordered = sorted(s.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    verdict = top if (ordered[0] >= 2.5 and margin >= 1.5) else "inconclusive"
    return {"scores": {k: round(v, 2) for k, v in s.items()}, "leaning": verdict,
            "note": "Heuristic tally of weak signals; not a probability."}
