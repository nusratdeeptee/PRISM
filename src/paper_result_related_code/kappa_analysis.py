"""Cohen's kappa between the Claude judge and human annotators (PRISM RQ2).

Usage:
  python kappa_analysis.py --claude-dir DIR --human-json human_annotations_25.json [--out results.csv] [--out-json results.json]

DIR holds {cond}_{meth}_25.csv for cond in s1, s2, mimicry_5, waterbird_test and
meth in idg, ig2 (columns: "Image name", "<feat> is present", "<feat> has attention").
human_annotations_25.json maps "cond|meth|annotator" -> {"image.png": [raw feature names]}
(the 25-image open-coding round -- the earlier 75-image round's file was
"human_annotations.json"; both --claude-dir's *_25.csv files and --human-json default to
the 25-image round now that it exists in results/rq2/).
Pair label: Claude = present AND attention; human = feature listed (names mapped to bank).

Writes two outputs: the per-(condition, method) kappa/agreement table as both a CSV
(--out, unchanged from the original 75-image version of this script) and a JSON file
(--out-json, new) that additionally includes the POOLED (per-method and overall) kappa
figures already computed in-console by summarize() but previously never persisted.
"""
import csv, sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths

ap = argparse.ArgumentParser()
ap.add_argument("--claude-dir", default=str(paths.RQ_RESULTS_DIR / "rq2"))
ap.add_argument("--human-json", default=str(paths.RQ_RESULTS_DIR / "rq2" / "human_annotations_25.json"))
ap.add_argument("--out", default=str(paths.RQ_RESULTS_DIR / "rq2" / "kappa_results.csv"))
ap.add_argument("--out-json", default=str(paths.RQ_RESULTS_DIR / "rq2" / "kappa_results.json"))
args = ap.parse_args()


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else paths.PRISM_ROOT / path


UP = str(_resolve(args.claude_dir)).rstrip("/") + "/"
H = {tuple(k.split("|")): v for k, v in json.load(open(_resolve(args.human_json))).items()}

# ---------- name -> bank mapping ----------
SG = {  # S1 / S2 bank
 "road":"road","lane in front and close":"road","lane in front":"road","adjacent lane":"road","adjacent lanes":"road",
 "double lane line":"road","lane lines":"road","adjacent lane lines":"road","road in front":"road",
 "vehicle in front":"vehicle","car infront":"vehicle","car in side lane":"vehicle","car in the side lane":"vehicle","car":"vehicle",
 "car ahead of the car infront":"vehicle","adjacent vehic":"vehicle","adjacent vehicle":"vehicle","vehicle in adjacent lane":"vehicle",
 "traffic light":"traffic light","traffic light post":"traffic light",
 "tree":"tree","trees":"tree",
 "sidewalk":"sidewalk","side walk at intersection":"sidewalk","sidewalk 5":"sidewalk",
 "streetlamp":"streetlamp","streetlight":"streetlamp","strretlamp":"streetlamp","lamp post base":"streetlamp",
 "building":"building","fence":"fence","sky":"sky"}
MIM = {
 "body":"animal","electric ray":"animal","fish":"animal","animal":"animal","white fish":"animal",
 "rocks":"rocks","rock":"rocks","white rock":"rocks",
 "sand":"sand","ground":"sand","white ground":"sand",
 "background":"background","unknown background":"background","black background":"background",
 "white background":"background","gray unknown background":"background",
 "white object":"white object"}
MIM_LIB = dict(MIM, **{"gray object":"white object","dark gray object":"white object","unknown object":"white object"})
WB = {
 "bird body":"bird","bird head":"bird","bird beak":"bird","bird":"bird",
 "background bird":"background birds","background birds":"background birds","goose":"background birds","duck":"background birds",
 "foliage":"foliage","trees":"foliage","leaves":"foliage","bamboo":"foliage","branch":"foliage","bush":"foliage","grass":"foliage",
 "ground":"ground","grounds":"ground","water":"water","sky":"sky",
 "person":"human","human face":"human","human head":"human","rock":"rocks","rocks":"rocks","fence":"fence"}
WB_STRICT = {k:v for k,v in WB.items() if k not in ("grass","goose","duck","bush")}
MAPS = {"s1":SG,"s2":SG,"mimicry_5":MIM,"waterbird_test":WB}

def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a)/n, sum(b)/n
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    if pe == 1: return float("nan"), po
    return (po - pe)/(1 - pe), po

def claude(cond, meth):
    rows = list(csv.DictReader(open(f"{UP}{cond}_{meth}_25.csv", newline="")))
    out = {}
    feats = sorted({k[:-len(" is present")] for k in rows[0] if k.endswith(" is present")})
    for r in rows:
        out[r["Image name"]] = {f: int(r[f+" is present"]) and int(r[f+" has attention"]) for f in feats}
    return out, feats

def run(maps=None, verbose=True):
    maps = maps or MAPS
    res = {}
    for cond in ["s1","s2","mimicry_5","waterbird_test"]:
        for meth in ["idg","ig2"]:
            C, feats = claude(cond, meth)
            M = H[(cond,meth,"annotator1")]; N = H[(cond,meth,"annotator2")]
            imgs = sorted(set(C) & set(M) & set(N))
            mp = maps[cond]
            def hum(d, img):
                got = {mp[n] for n in d[img] if n in mp}
                return {f: int(f in got) for f in feats}
            cl, ma, nu, un = [], [], [], []
            for i in imgs:
                m, n = hum(M, i), hum(N, i)
                for f in feats:
                    cl.append(C[i][f]); ma.append(m[f]); nu.append(n[f]); un.append(int(m[f] or n[f]))
            res[(cond,meth)] = dict(n_img=len(imgs), n_pairs=len(cl), cl=cl, ma=ma, nu=nu, un=un,
                                    missing=sorted(set(C)-set(M)|set(C)-set(N)))
    return res

ROWS = []
POOLED_SUMMARY = {}  # label -> {method -> {n_pairs, kappa_*, agree_*}}, filled by summarize()
def summarize(res, label):
    print(f"\n===== {label} =====")
    print(f"{'cond':16}{'meth':5}{'imgs':>5}{'pairs':>6} | {'C-M k':>7}{'C-N k':>7}{'M-N k':>7}{'C-U k':>7} | {'C-M %':>6}{'C-N %':>6}{'M-N %':>6}")
    pooled = {m: dict(cl=[],ma=[],nu=[],un=[]) for m in ("idg","ig2","all")}
    for (c, m), r in res.items():
        ck, cp = kappa(r["cl"], r["ma"]); nk, np_ = kappa(r["cl"], r["nu"])
        mk, mp_ = kappa(r["ma"], r["nu"]); uk, _ = kappa(r["cl"], r["un"])
        ROWS.append([c,m,r["n_img"],r["n_pairs"],round(ck,3),round(nk,3),round(mk,3),round(uk,3),round(100*cp,1),round(100*np_,1),round(100*mp_,1)])
        print(f"{c:16}{m:5}{r['n_img']:5}{r['n_pairs']:6} | {ck:7.2f}{nk:7.2f}{mk:7.2f}{uk:7.2f} | {100*cp:6.1f}{100*np_:6.1f}{100*mp_:6.1f}")
        for k in ("cl","ma","nu","un"):
            pooled[m][k] += r[k]; pooled["all"][k] += r[k]
    POOLED_SUMMARY[label] = {}
    for m in ("idg","ig2","all"):
        p = pooled[m]
        ck, cp = kappa(p["cl"], p["ma"]); nk, np_ = kappa(p["cl"], p["nu"]); mk, mp_ = kappa(p["ma"], p["nu"]); uk,_ = kappa(p["cl"], p["un"])
        print(f"{'POOLED':16}{m:5}{'':5}{len(p['cl']):6} | {ck:7.2f}{nk:7.2f}{mk:7.2f}{uk:7.2f} | {100*cp:6.1f}{100*np_:6.1f}{100*mp_:6.1f}")
        POOLED_SUMMARY[label][m] = dict(
            n_pairs=len(p["cl"]),
            kappa_claude_annotator1=round(ck, 3), kappa_claude_annotator2=round(nk, 3),
            kappa_annotator1_annotator2=round(mk, 3), kappa_claude_union=round(uk, 3),
            agree_claude_annotator1_pct=round(100 * cp, 1), agree_claude_annotator2_pct=round(100 * np_, 1),
            agree_annotator1_annotator2_pct=round(100 * mp_, 1),
        )
    return pooled

HEADER = ["cond","method","images","pairs","kappa_claude_annotator1","kappa_claude_annotator2","kappa_annotator1_annotator2","kappa_claude_union","agree_claude_annotator1","agree_claude_annotator2","agree_annotator1_annotator2"]

if __name__ == "__main__":
    out_csv = _resolve(args.out)
    out_json = _resolve(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    r = run()
    for k, v in r.items():
        if v["missing"]: print("images without human labels in", k, v["missing"])
    summarize(r, "primary mapping")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(ROWS)
    print("wrote", out_csv)

    # JSON mirror of the CSV just written (per-condition rows, "primary mapping" only --
    # the sensitivity-analysis reruns below append more rows to ROWS/POOLED_SUMMARY, but
    # those stay console-only, same as the CSV above only covers "primary mapping") plus
    # the POOLED (per-method and overall) kappa figures summarize() already computes but
    # previously only printed.
    payload = {
        "per_condition": [dict(zip(HEADER, row)) for row in ROWS],
        "pooled": POOLED_SUMMARY["primary mapping"],
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", out_json)
    # sensitivity: liberal Mimicry mapping, strict Waterbirds mapping
    s = dict(MAPS); s["mimicry_5"] = MIM_LIB
    summarize(run(s), "sensitivity: liberal Mimicry white-object mapping")
    s = dict(MAPS); s["waterbird_test"] = WB_STRICT
    summarize(run(s), "sensitivity: grass/bush/goose/duck left unmapped (Waterbirds)")
    # unmapped names report
    print("\nUnmapped human names (ignored, not in bank):")
    for (cond, meth, who), d in H.items():
        mp = MAPS[cond]
        um = sorted({n for v in d.values() for n in v if n not in mp})
        print(f"  {cond}/{meth}/{who}: {', '.join(um)}")
