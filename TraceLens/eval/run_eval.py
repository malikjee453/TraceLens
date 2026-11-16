"""Measure how often TraceLens is right before you trust it.

Folder layout (put 50+ files in each; mix sources, sizes, platforms, and recompress some):
    eval/data/real/        photos from real cameras/phones
    eval/data/ai/          images from AI generators
    eval/data/graphic/     illustrations, renders, posters, screenshots

Run:  GROQ_API_KEY=gsk_... python eval/run_eval.py eval/data [--no-ai] [--model qwen/qwen3.8-27b]
"""
import argparse, collections, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracelens.ai import DEFAULT_MODEL, AIError, analyse_with_groq
from tracelens.imaging import analyze_image
from tracelens.signals import image_signals, leaning

TRUTH = {"real": "camera_capture", "ai": "ai_generated", "graphic": "illustration_or_graphic"}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--no-ai", action="store_true"); ap.add_argument("--model", default=DEFAULT_MODEL)
a = ap.parse_args()
key = os.environ.get("GROQ_API_KEY", "")
if not a.no_ai and not key:
    sys.exit("Set GROQ_API_KEY or use --no-ai")

conf = collections.Counter(); n = 0; wrong_conf = []
for folder, truth in TRUTH.items():
    for p in sorted((Path(a.root) / folder).glob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        res, im, _ = analyze_image(p.read_bytes(), p.name)
        sig = image_signals(res); lean = leaning(sig)
        if a.no_ai:
            pred = {"camera": "camera_capture", "generated": "ai_generated", "graphic": "illustration_or_graphic",
                    "edited": "edited_or_composite"}.get(lean["leaning"], "uncertain"); c = 0
        else:
            try:
                v = analyse_with_groq(key, a.model, [im], res, sig, lean, "image"); pred, c = v["category"], v["confidence"]
            except AIError as e:
                print("ERR", p.name, e); pred, c = "error", 0
            time.sleep(1.0)  # be gentle with rate limits
        conf[(truth, pred)] += 1; n += 1
        if pred != truth and c >= 80:
            wrong_conf.append((p.name, truth, pred, c))
        print(f"{p.name:40s} truth={truth:24s} pred={pred:24s} conf={c}")

preds = sorted({k[1] for k in conf}); print("\nConfusion matrix (rows=truth, cols=prediction)")
print(f"{'':26s}" + "".join(f"{x[:14]:>16s}" for x in preds))
for t in TRUTH.values():
    print(f"{t:26s}" + "".join(f"{conf[(t, x)]:16d}" for x in preds))
ok = sum(conf[(t, t)] for t in TRUTH.values()); unc = sum(v for (t, p), v in conf.items() if p == "uncertain")
print(f"\nAccuracy: {ok}/{n} = {ok / max(n, 1):.1%} | 'uncertain' rate: {unc / max(n, 1):.1%}")
print(f"Confident mistakes (conf>=80): {len(wrong_conf)}")
for w in wrong_conf: print("  ", w)
