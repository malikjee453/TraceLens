from __future__ import annotations

import base64
import datetime
import html
import io
import json

from PIL import Image

from . import APP_NAME, VERSION
from .ai import CATEGORY_LABELS

E = html.escape


def _b64(im: Image.Image, max_side=900) -> str:
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return base64.b64encode(b.getvalue()).decode()


def _ul(items):
    return "<ul>" + "".join(f"<li>{E(str(i))}</li>" for i in items) + "</ul>" if items else "<p><i>none</i></p>"


def build_json(res: dict, signals: list, lean: dict, verdict: dict | None) -> dict:
    out = {"application": APP_NAME, "application_version": VERSION,
           "report_generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "measurements": res, "signals": signals, "heuristic_tally": lean,
           "ai_assessment": verdict,
           "disclaimer": "Investigative triage aid, not proof. Not validated for legal or disciplinary use."}
    return out


def build_html(res: dict, signals: list, lean: dict, verdict: dict | None,
               preview: Image.Image | None, ela_img: Image.Image | None) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    v_html = "<p><i>AI assessment was not run (local-only mode).</i></p>"
    if verdict:
        v_html = (f"<h2>Assessment: {E(CATEGORY_LABELS[verdict['category']])} "
                  f"<small>(confidence {verdict['confidence']}/100, {E(str(verdict.get('stage', '')))} pass)</small></h2><p>{E(verdict['summary'])}</p>"
                  + (f"<p><b>Method guess:</b> {E(verdict['generation_method_guess'])}</p>" if verdict.get("generation_method_guess") else "")
                  + "<h3>Visual observations</h3>" + _ul(verdict["visual_observations"])
                  + "<h3>Evidence for camera capture</h3>" + _ul(verdict["evidence_for_camera"])
                  + "<h3>Evidence for synthetic origin</h3>" + _ul(verdict["evidence_for_synthetic"])
                  + "<h3>Edit indicators</h3>" + _ul(verdict["edit_indicators"])
                  + "<h3>Unknown / missing</h3>" + _ul(verdict["unknown_or_missing"])
                  + "<h3>Limitations</h3>" + _ul(verdict["limitations"])
                  + "<h3>What would change this assessment</h3>" + _ul(verdict["what_would_change_my_mind"])
                  + "<h3>Recommended next steps</h3>" + _ul(verdict["recommended_next_steps"])
                  + ("<h3>Guardrail notes</h3>" + _ul(verdict["guardrail_notes"]) if verdict.get("guardrail_notes") else "")
                  + f"<p><small>Model: {E(str(verdict.get('_meta', {}).get('model', '?')))}</small></p>")
    sig_rows = "".join(f"<tr><td>{E(s['label'])}</td><td>{E(s['direction'])}</td><td>{s['weight']}</td><td>{E(s['why'])}</td></tr>" for s in signals)
    meas = {k: v for k, v in res.items() if k != "exif"}
    exif_rows = "".join(f"<tr><th>{E(str(k))}</th><td>{E(str(v))}</td></tr>" for k, v in res.get("exif", {}).items()
                        if not str(k).startswith("GPS."))
    imgs = ""
    if preview is not None:
        imgs += f'<div><h3>Evidence preview</h3><img src="data:image/jpeg;base64,{_b64(preview)}"></div>'
    if ela_img is not None:
        imgs += f'<div><h3>ELA (x10)</h3><img src="data:image/jpeg;base64,{_b64(ela_img)}"></div>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{APP_NAME} report</title>
<style>body{{font:14px Arial,sans-serif;margin:32px;color:#172033;max-width:1000px}}h1{{color:#17365d}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccd3df;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#edf2f8}}pre{{white-space:pre-wrap;word-break:break-word;margin:0;font-size:12px}}
.w{{background:#fff4d6;padding:12px;border-left:4px solid #d99b00}}img{{max-width:100%;border:1px solid #ccd3df}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}</style></head><body>
<h1>{APP_NAME} — Examination report</h1><p>Generated {now} · v{VERSION}</p>
<div class="w"><b>Interpretation warning:</b> investigative aid, not a certification of authenticity. Automated
indicators and AI judgements can be wrong, especially on new generators and compressed media. Corroborate independently.</div>
<h2>File</h2><table><tr><th>Name</th><td>{E(res['filename'])}</td></tr><tr><th>SHA-256</th><td><code>{res['sha256']}</code></td></tr>
<tr><th>Size</th><td>{res['size_bytes']} bytes</td></tr></table>
{v_html}<div class="grid">{imgs}</div>
<h2>Measured signals (weak heuristics)</h2><table><tr><th>Signal</th><th>Points to</th><th>Weight</th><th>Detail</th></tr>{sig_rows}</table>
<p>Tally: <code>{E(json.dumps(lean))}</code></p>
<h2>Measurements</h2><pre>{E(json.dumps(meas, indent=1, default=str))}</pre>
<h2>EXIF (GPS omitted)</h2><table>{exif_rows or '<tr><td>No EXIF fields.</td></tr>'}</table></body></html>"""
