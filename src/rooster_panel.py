"""Random failed images of one ImageNet class per technique + IDG attention overlay -> results/<tag>_panel.png
usage: python src/rooster_panel.py [class=7] [n_per_technique=1] [seed=0]
(PINNED/EXCLUDE keep the earlier rooster picks; they apply only to class 7 with n=1.)"""
import sys, random, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CLS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TAG = {7: "rooster", 1: "goldfish", 8: "hen", 5: "electric_ray"}.get(CLS, f"class{CLS}") + (f"_n{N}" if N > 1 else "") + (f"_seed{SEED}" if SEED else "")
TECH = {"Mimicry": f"mimicry_{CLS}", "GenAI-TIG": f"biggan_ai_tigs_{CLS}", "EP-DNN": f"ep_dnn_{CLS}"}
legacy = CLS == 7 and N == 1
PINNED = {"GenAI-TIG": 780, "EP-DNN": 911} if legacy else {}
EXCLUDE = {"Mimicry": {675}} if legacy else {}
rng = random.Random(SEED)
plt.rcParams.update({"font.weight": "bold", "axes.titleweight": "bold"})

picks = {}  # name -> list of (iid, img, map)
for name, d in TECH.items():
    df = pd.read_csv(ROOT / "data" / d / "metadata.csv")
    fails = df[df.label == "fail"]
    if fails.empty:
        if name != "EP-DNN":
            raise SystemExit(f"{name}: no failures in {d}")
        print(f"{name}: no failures in {d}; using a random pass image"); fails = df
    if name in PINNED:
        ids = [PINNED[name]]
    else:
        pool = [i for i in fails.image_id.tolist() if i not in EXCLUDE.get(name, ())]
        if len(pool) < N: raise SystemExit(f"{name}: only {len(pool)} failures")
        ids = rng.sample(pool, N)
    picks[name] = []
    for iid in ids:
        img = Image.open(ROOT / "data" / d / "images" / f"{iid}.png").convert("RGB").resize((224, 224))
        a = np.load(ROOT / "output" / d / "wide_resnet50_2" / "idg" / f"{iid}_attr.npy")
        m = np.abs(a).sum(0); m = (m - m.min()) / (m.max() - m.min() + 1e-12)
        picks[name].append((iid, np.asarray(img), m))
    print(name, "image_ids", ids)

ncol = 3 * N
fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 6), gridspec_kw=dict(wspace=0, hspace=0), squeeze=False)
j = 0
for name, items in picks.items():
    for k, (iid, img, m) in enumerate(items):
        axes[0, j].imshow(img, aspect="auto")
        axes[1, j].imshow(img, aspect="auto"); axes[1, j].imshow(m, cmap="jet", alpha=0.5, aspect="auto")
        j += 1
for ax in axes.ravel(): ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
out = ROOT / "results" / f"{TAG}_panel.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0); print(out)
