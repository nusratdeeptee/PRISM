"""RQ3 result aggregation: collects, for every base model (ERM) and its repaired versions, the test
metrics the DISC / B2T papers report (worst-group accuracy, average accuracy, and ROC AUC for ISIC)
into one JSON file: results/rq3_repair_metrics.json. All values are in PERCENT (0-100).

Repair methods / datasets / arms (whatever exists on disk is recorded; missing runs are null):
  DISC : waterbird (CUB, ERM seeds 0-2), metashift (Cat/Dog, seeds 0-2), isic (trap-sets 1-5)
         arms: erm | disc (DISC's own concept bank) | disc_prism (PRISM concept bank)
  B2T  : waterbird (seeds 0-2), celeba (seeds 0-2)
         arms: erm | b2t (B2T's own discovered keywords, CLIP>1.0) | b2t_prism (PRISM keywords, CLIP>1.0)

Where each number comes from (never recomputed from a different selection rule):
  DISC runs   : the run's log.txt ends with the epoch selected on VALIDATION data (pandas "Name: <row>"
                block); the test metrics are read from that row of the run's test.csv (all columns, so
                worst_group_acc / avg_acc / roc_auc are all available) and cross-checked against the
                avg_acc printed in the log. ISIC selects on val ROC AUC, the others on val worst-group.
  B2T runs    : the "Best ... (test) at epoch E" lines at the end of the SLURM .out file (epoch chosen on
                validation worst-group accuracy by B2T's own code).
  Sagawa CelebA ERM (3 seeds): per-epoch val.csv / test.csv only. Selected epoch = best val avg_acc
                (the Sagawa-code protocol); worst-group = min over the 4 (Blond, Male) groups. The
                val-worst-group-selected numbers are also stored under "alt_selection_val_worst_group".
  B2T CelebA ERM (ours): seed 0 only, from slurm/celeba/erm_celeba.out.
  B2T Waterbird ERM: the DISC/CUB ERM runs (seeds 0-2) -- the same checkpoints B2T/DISC start from.

Per (dataset, method, arm) a "summary" gives mean and sample sd (ddof=1) over the seeds available.

Usage:
    python rq3.py            # writes results/rq3_repair_metrics.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "output" / "model_repair"
SLURM = ROOT / "slurm"
SAGAWA_CELEBA = ROOT.parent / "baseline" / "group_DRO_sagawa" / "output"
OUT = ROOT / "results" / "rq3_repair_metrics.json"

METRIC_COLS = {"worst_group_acc": "worst_group_acc", "avg_acc": "avg_acc", "roc_auc": "roc_auc"}


def _pct(x):
    return None if x is None or pd.isna(x) else round(float(x) * 100, 4)


# ------------------------------------------------------------------ DISC ------------------
def read_disc_run(run_dir: Path):
    """Test metrics at the val-selected epoch of a DISC-code run (log.txt + test.csv)."""
    log, test = run_dir / "log.txt", run_dir / "test.csv"
    if not (log.exists() and test.exists()):
        return None
    txt = log.read_text(errors="ignore")
    names = re.findall(r"Name:\s*(\d+),\s*dtype", txt)
    if not names:                                    # run never reached its final selection printout
        return {"status": "incomplete (no selected-epoch block in log.txt)"}
    row_idx = int(names[-1])
    df = pd.read_csv(test)
    row = df.iloc[row_idx]
    rec = {"epoch": int(row["epoch"]) if "epoch" in df else row_idx}
    for k, col in METRIC_COLS.items():
        if col in df:
            rec[k] = _pct(row[col])
    # cross-check against the avg_acc the log itself printed for that row
    m = re.findall(r"avg_acc\s+([0-9.]+)\s*\n", txt.split("Name:")[-2] if "Name:" in txt else "")
    if m and abs(float(m[-1]) * 100 - rec["avg_acc"]) > 1e-3:
        rec["warning"] = f"log avg_acc {float(m[-1])*100:.4f} != test.csv row {rec['avg_acc']:.4f}"
    rec["run_dir"] = str(run_dir.relative_to(ROOT))
    return rec


ERM_W = "reweight_groups=0-augment=0-lr=0.001-batch_size=32-n_epochs=300-seed={s}"
DISC_W = "color-texture-nature-n_concept_imgs=200-n_clusters=3-reweight_groups=1-augment=1-lr=0.0001-batch_size=32-n_epochs=10-seed={d}"
PRISM_W = "{p}-n_concept_imgs=200-n_clusters=3-reweight_groups=1-augment=1-lr=0.0001-batch_size=32-n_epochs=10-seed={d}"
ERM_M = "reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-seed={s}"
DISC_M = "nature-city-color-texture-household-others-n_concept_imgs=200-n_clusters=2-reweight_groups=0-augment=1-lr=0.0005-batch_size=16-n_epochs=100-seed={s}"
PRISM_M = "grit_prism_metashift_s{s}-n_concept_imgs=200-n_clusters=2-reweight_groups=0-augment=1-lr=0.0005-batch_size=16-n_epochs=100-seed={s}"
ERM_I = "reweight_groups=0-augment=0-lr=0.001-batch_size=16-n_epochs=100-trapset_id={s}"
DISC_I = "color-texture-n_concept_imgs=200-n_clusters=3-reweight_groups=0-augment=0-lr=0.0005-batch_size=16-n_epochs=100-trapset_id={s}"
PRISM_I = "grit_prism_isic_trapset{s}-n_concept_imgs=200-n_clusters=3-reweight_groups=0-augment=0-lr=0.0005-batch_size=16-n_epochs=100-trapset_id={s}"


def disc_section():
    out = {}
    # Waterbirds: the seed-0 ERM is repaired under DISC's fixed seed 5 (see baseline/DISC/scripts/waterbirds.sh);
    # ERM seeds 1, 2 are repaired under seeds 1, 2.
    wb = {}
    for s in (0, 1, 2):
        d = 5 if s == 0 else s
        pname = "grit_prism_waterbird" if s == 0 else f"grit_prism_waterbird_s{s}"
        wb[str(s)] = {
            "erm": read_disc_run(REPAIR / "DISC/CUB/ERM" / ERM_W.format(s=s)),
            "disc": read_disc_run(REPAIR / "DISC/CUB/DISC" / DISC_W.format(d=d)),
            "disc_prism": read_disc_run(REPAIR / "DISC/CUB/DISC" / PRISM_W.format(p=pname, d=d)),
        }
    out["waterbird"] = wb
    out["metashift"] = {str(s): {
        "erm": read_disc_run(REPAIR / "DISC/MetaDatasetCatDog/ERM" / ERM_M.format(s=s)),
        "disc": read_disc_run(REPAIR / "DISC/MetaDatasetCatDog/DISC" / DISC_M.format(s=s)),
        "disc_prism": read_disc_run(REPAIR / "DISC/MetaDatasetCatDog/DISC" / PRISM_M.format(s=s)),
    } for s in (0, 1, 2)}
    out["isic"] = {f"trapset{s}": {
        "erm": read_disc_run(REPAIR / "DISC/ISIC/ERM" / ERM_I.format(s=s)),
        "disc": read_disc_run(REPAIR / "DISC/ISIC/DISC" / DISC_I.format(s=s)),
        "disc_prism": read_disc_run(REPAIR / "DISC/ISIC/DISC" / PRISM_I.format(s=s)),
    } for s in range(1, 6)}
    return out


# ------------------------------------------------------------------ B2T -------------------
def read_b2t_out(path: Path):
    """'Best ... (test) at epoch E' lines of a B2T group_dro SLURM .out file."""
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    g = re.findall(r"Best worst group accuracy \(test\) at epoch (\d+): ([0-9.]+)", txt)
    a = re.findall(r"Best average accuracy \(test\) at epoch (\d+): ([0-9.]+)", txt)
    if not g or not a:
        return {"status": "incomplete (no final 'Best ... (test)' lines)"}
    return {"epoch": int(g[-1][0]), "worst_group_acc": round(float(g[-1][1]), 4),
            "avg_acc": round(float(a[-1][1]), 4), "source": str(path.relative_to(ROOT))}


def read_sagawa_erm(seed: int):
    d = SAGAWA_CELEBA / f"celeba_erm_seed{seed}"
    if not (d / "val.csv").exists():
        return None
    v, t = pd.read_csv(d / "val.csv"), pd.read_csv(d / "test.csv")
    grp = [f"avg_acc_group:{i}" for i in range(4)]
    v["_wg"], t["_wg"] = v[grp].min(axis=1), t[grp].min(axis=1)

    def at(i):
        return {"epoch": int(t.loc[i, "epoch"]), "worst_group_acc": _pct(t.loc[i, "_wg"]),
                "avg_acc": _pct(t.loc[i, "avg_acc"])}
    rec = at(v["avg_acc"].idxmax())
    rec["selection"] = "best val avg_acc (Sagawa protocol)"
    rec["alt_selection_val_worst_group"] = at(v["_wg"].idxmax())
    rec["source"] = str(d.relative_to(SAGAWA_CELEBA.parent.parent.parent))
    return rec


def b2t_section():
    out = {"waterbird": {}, "celeba": {}}
    for s in (0, 1, 2):
        out["waterbird"][str(s)] = {
            "erm": read_disc_run(REPAIR / "DISC/CUB/ERM" / ERM_W.format(s=s)),
            "b2t": read_b2t_out(SLURM / f"waterbird/b2t_discovered_clip1_seed{s}.out"),
            "b2t_prism": read_b2t_out(SLURM / f"waterbird/b2t_prism_clip1_seed{s}.out"),
        }
        out["celeba"][str(s)] = {
            "erm": read_sagawa_erm(s),
            "b2t": read_b2t_out(SLURM / f"celeba/b2t_discovered_clip1_seed{s}.out"),   # only seed 1 exists (others: no keyword >1.0)
            "b2t_prism": read_b2t_out(SLURM / f"celeba/b2t_prism_clip1_seed{s}.out"),
        }
    out["celeba"]["erm_b2t_pipeline_seed0"] = {"erm": read_b2t_out(SLURM / "celeba/erm_celeba.out")}
    return out


# ------------------------------------------------------------------ summary ---------------
def summarize(block):
    """mean / sample sd of every metric per arm over the seeds that have that arm."""
    arms = {a for runs in block.values() for a in runs}
    summ = {}
    for arm in sorted(arms):
        recs = [r[arm] for r in block.values() if r.get(arm) and "status" not in r[arm]]
        s = {"n_seeds": len(recs)}
        for m in ("worst_group_acc", "avg_acc", "roc_auc"):
            vals = [x[m] for x in recs if x.get(m) is not None]
            if vals:
                s[m] = {"mean": round(float(np.mean(vals)), 4),
                        "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None}
        summ[arm] = s
    return summ


def main():
    result = {"_meta": {
        "units": "percent (0-100), test set, epoch chosen on validation data",
        "note": "Per-seed values under 'runs'; 'summary' = mean / sample sd over seeds (ISIC 'seeds' are trap-set ids).",
        "isic_note": "ISIC selects on val ROC AUC; worst_group_acc there = min over hair=0/1.",
    }}
    for method, sec in (("DISC", disc_section()), ("B2T", b2t_section())):
        result[method] = {}
        for ds, block in sec.items():
            main_block = {k: v for k, v in block.items() if not k.startswith("erm_b2t")}
            result[method][ds] = {"runs": block, "summary": summarize(main_block)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    for method in ("DISC", "B2T"):
        for ds, v in result[method].items():
            for arm, s in v["summary"].items():
                f = lambda m: (f"{s[m]['mean']:.2f}" + (f"±{s[m]['sd']:.2f}" if s[m]['sd'] is not None else "")) if m in s else "-"
                print(f"{method:5s} {ds:10s} {arm:11s} n={s['n_seeds']}  WGA {f('worst_group_acc'):>12s}  avg {f('avg_acc'):>12s}  AUC {f('roc_auc'):>12s}")


if __name__ == "__main__":
    main()
