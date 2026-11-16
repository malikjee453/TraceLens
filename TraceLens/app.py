import hmac
import io
import json
import os

import streamlit as st
from PIL import Image

from tracelens import APP_NAME, VERSION
from tracelens.ai import CATEGORY_LABELS, DEFAULT_MODEL, AIError, analyse_with_groq
from tracelens.imaging import analyze_image
from tracelens.report import build_html, build_json
from tracelens.signals import image_signals, leaning, video_signals
from tracelens.video import analyze_video

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
ICON = {"camera_capture": "📷", "ai_generated": "🤖", "illustration_or_graphic": "🎨",
        "edited_or_composite": "✂️", "uncertain": "❔"}


def secret(name, default=""):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:           # no secrets.toml present
        return os.environ.get(name, default)


# ------------------------------------------------------------------ optional password gate
pw = secret("APP_PASSWORD")
if pw and not st.session_state.get("authed"):
    st.title(f"🔎 {APP_NAME}")
    typed = st.text_input("Password", type="password")
    if typed and hmac.compare_digest(typed.encode(), str(pw).encode()):
        st.session_state["authed"] = True
        st.rerun()
    elif typed:
        st.error("Wrong password.")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    use_ai = st.toggle("Use Groq AI assessment", value=True,
                       help="Off = fully local measurements only. On = the image (downscaled) and a metadata "
                            "summary are sent to Groq. GPS coordinates are never sent.")
    key = secret("GROQ_API_KEY")
    if not key:
        key = st.text_input("Groq API key", type="password", help="Kept only in this browser session.")
    else:
        st.success("Groq key loaded from secrets")
    model = st.text_input("Vision model", value=secret("GROQ_VISION_MODEL", DEFAULT_MODEL),
                          help="Must be a Groq model with image input and JSON mode.")
    n_frames = st.slider("Video frames to sample", 3, 12, 6)
    st.divider()
    st.caption("🔒 **Privacy:** files are processed in memory and are not saved by this app. With AI on, a "
               "downscaled copy goes to Groq. Do not upload sensitive evidence to a public deployment.")
    st.caption(f"v{VERSION}")

st.title(f"🔎 {APP_NAME}")
st.caption("Camera capture vs. computer-generated: measured evidence + an evidence-grounded vision-LLM review. "
           "A triage aid, not proof.")

up = st.file_uploader("Select an evidence file", type=IMAGE_TYPES + VIDEO_TYPES)


def run(file) -> dict:
    data = file.getvalue()
    ext = file.name.rsplit(".", 1)[-1].lower()
    is_video = ext in VIDEO_TYPES
    with st.status("Examining…", expanded=True) as status:
        status.write("Hashing and measuring")
        if is_video:
            res, frames = analyze_video(data, file.name, n_frames=n_frames)
            signals, preview, ela_img, ai_images = video_signals(res), None, None, frames
        else:
            res, im, ela_img = analyze_image(data, file.name)
            signals, preview, ai_images = image_signals(res), im, [im]
        lean = leaning(signals)
        verdict, ai_error = None, None
        if use_ai:
            status.write(f"Asking Groq ({model})")
            try:
                verdict = analyse_with_groq(key, model, ai_images, res, signals, lean, res["media_type"])
            except AIError as e:
                ai_error = str(e)
        status.update(label="Done", state="complete", expanded=False)
    return {"res": res, "signals": signals, "lean": lean, "verdict": verdict, "ai_error": ai_error,
            "preview": preview, "ela": ela_img, "frames": ai_images if is_video else []}


if up is not None:
    sig = (up.name, up.size)
    if st.session_state.get("sig") != sig or st.session_state.get("ai_flag") != (use_ai, model):
        try:
            st.session_state["out"] = run(up)
            st.session_state["sig"], st.session_state["ai_flag"] = sig, (use_ai, model)
        except Exception as e:
            st.session_state.pop("out", None)
            st.error(f"Analysis failed: {e}")
    out = st.session_state.get("out")
    if out:
        res, v = out["res"], out["verdict"]
        left, right = st.columns([1, 1])
        with left:
            if out["preview"] is not None:
                st.image(out["preview"], caption=res["filename"], use_container_width=True)
            for i, f in enumerate(out["frames"][:6]):
                st.image(f, caption=f"Sampled frame {i + 1}", use_container_width=True)
        with right:
            if v:
                st.subheader(f"{ICON[v['category']]} {CATEGORY_LABELS[v['category']]}")
                st.progress(v["confidence"] / 100, text=f"Confidence {v['confidence']}/100 (calibrated, capped)")
                st.write(v["summary"])
                if v.get("generation_method_guess"):
                    st.info(v["generation_method_guess"])
                for n in v.get("guardrail_notes", []):
                    st.warning(n)
            elif out["ai_error"]:
                st.error(out["ai_error"])
                st.caption("Local measurements below are still valid.")
            else:
                st.info("AI assessment is off. Showing local measurements only.")
            lean = out["lean"]
            st.metric("Local heuristic leaning", lean["leaning"],
                      help="Tally of weak signals; shown for transparency, not a verdict.")
            st.caption(f"SHA-256: `{res['sha256']}`")

        if v:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Evidence for camera capture**")
                st.write("\n".join(f"- {x}" for x in v["evidence_for_camera"]) or "—")
            with c2:
                st.markdown("**Evidence for synthetic origin**")
                st.write("\n".join(f"- {x}" for x in v["evidence_for_synthetic"] + v["edit_indicators"]) or "—")
            with c3:
                st.markdown("**Unknown / limitations**")
                st.write("\n".join(f"- {x}" for x in v["unknown_or_missing"] + v["limitations"]) or "—")
            with st.expander("Visual observations, what would change the verdict, next steps"):
                for title, k in (("Visual observations", "visual_observations"),
                                 ("What would change this assessment", "what_would_change_my_mind"),
                                 ("Recommended next steps", "recommended_next_steps")):
                    st.markdown(f"**{title}**")
                    st.write("\n".join(f"- {x}" for x in v[k]) or "—")

        st.subheader("Measured signals")
        st.caption("Weak, documented heuristics. Missing EXIF is never treated as evidence of fakery.")
        st.dataframe([{"Signal": s["label"], "Points to": s["direction"], "Weight": s["weight"], "Detail": s["why"]}
                      for s in out["signals"]], use_container_width=True, hide_index=True)
        if out["ela"] is not None:
            with st.expander("ELA image (JPEG only)"):
                st.image(out["ela"], use_container_width=True)
                st.caption(res["ela"].get("note", ""))
        with st.expander("Raw measurements and EXIF"):
            st.json({k: v_ for k, v_ in res.items() if k != "exif"})
            st.json({k: v_ for k, v_ in res["exif"].items() if not str(k).startswith("GPS.")} if "exif" in res else {})

        html_report = build_html(res, out["signals"], out["lean"], v, out["preview"] or (out["frames"][0] if out["frames"] else None), out["ela"])
        json_report = json.dumps(build_json(res, out["signals"], out["lean"], v), indent=2, default=str)
        d1, d2 = st.columns(2)
        stem = res["filename"].rsplit(".", 1)[0][:50]
        d1.download_button("⬇️ HTML report", html_report, f"{stem}_report.html", "text/html")
        d2.download_button("⬇️ JSON report", json_report, f"{stem}_report.json", "application/json")
else:
    st.info("Upload an image or video to begin.")
