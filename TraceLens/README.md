# TraceLens Forensics 2.0 (Streamlit + Groq)

Camera capture vs. computer-generated (AI / illustration / render) triage for images and videos.

**How it works**
1. **Measure (local):** SHA-256, full EXIF (IFD0 + Exif + GPS sub-IFDs), embedded EXIF thumbnail, sensor shot-noise and demosaicing statistics, C2PA/XMP/IPTC "digital source type" and AI-tool string scan, PNG prompt chunks, JPEG quantisation tables, ELA, noise residual, FFT spectrum, flat-colour/stylisation stats, generator-size hints. Video: ffprobe container tags, sampled frames, short-burst temporal stats (flicker, noise repetition).
2. **Reason (Groq vision LLM):** the image/frames plus the measurements go to a forensics prompt with an explicit taxonomy (`camera_capture`, `ai_generated`, `illustration_or_graphic`, `edited_or_composite`, `uncertain`), anti-over-claiming rules, and JSON output.
3. **Two stages:** a fast pass accepts clearly real camera media immediately (confidence >= 80 after guardrails). Anything else gets a deep pass on native-resolution crops with more reasoning; if the two passes disagree the result is `uncertain`.
4. **Guardrails (code, not prompt):** declared AI metadata overrides the model; the ceiling for a "real camera" verdict rises with measured support: 70 (visual only), 78/85 (sensor-noise statistics), 90/95 (camera metadata); a graphic/synthetic signal holds it at 70; visual-only AI claims are capped at 80; contradictions downgrade to `uncertain`.

> This is a triage aid, **not** a validated forensic tool. Modern generators fool both people and models. Run `eval/run_eval.py` on your own labelled files before trusting it.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repo and upload this folder's contents (GitHub → *Add file → Upload files*, or `git init && git add . && git commit -m "TraceLens 2.0" && git remote add origin <url> && git push -u origin main`).
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`. Under *Advanced settings* choose Python 3.11 or 3.12.
3. In *Advanced settings → Secrets* paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   APP_PASSWORD = "choose-a-password"   # recommended: public apps let strangers spend your Groq quota
   ```
4. Deploy. `packages.txt` installs ffmpeg/ffprobe for video metadata.

Never commit your real key (`.streamlit/secrets.toml` is git-ignored). If a key leaks, revoke it in the Groq console.

## Run locally
```
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit
streamlit run app.py
```

## Model
Default `qwen/qwen3.8-27b` (Groq vision, JSON mode, max 3 images/request). It is a *preview* model; if Groq retires it, set `GROQ_VISION_MODEL` in secrets or edit the sidebar field. Check https://console.groq.com/docs/vision for current vision models.

## Privacy
Files are handled in memory and not stored by the app. With AI on, a downscaled image and a metadata summary are sent to Groq (GPS coordinates and file paths are withheld). Use the sidebar toggle for local-only measurements.

## Known limits
- C2PA is detected, **not cryptographically verified**.
- Invisible watermarks (e.g. SynthID) are not detected.
- Video analysis uses sampled frames and one short burst, not every frame; no face-swap detector is included.
- Sensor-noise thresholds were tuned on synthetic data only; calibrate with `eval/run_eval.py`. Photos stripped by WhatsApp/social media top out around 85 by design.
- ELA, noise and spectrum are weak and break under resizing/recompression.

## Evaluate
```
eval/data/real/  eval/data/ai/  eval/data/graphic/     # 50+ files each
GROQ_API_KEY=gsk_... python eval/run_eval.py eval/data
```
Prints a confusion matrix, accuracy, uncertain-rate and confident mistakes.

License: MIT (application source). Models and third-party libraries keep their own licenses.
