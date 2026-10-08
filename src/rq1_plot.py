import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/rq1_feature_counts.json"))
# rq1.py now also writes a "branch" field ("balanced"/"imbalanced"/None) alongside
# "total"/"validated" at every (dataset, attention, n) entry -- unused here (this plot only
# tracks feature counts), pooled() below reads "total"/"validated" by explicit key name so
# the extra field is harmless.
fam = {"SGSM": lambda k: k.startswith("S") and k[1:].isdigit(),
       "Mimicry": lambda k: k.startswith("mimicry"),
       "GenAI-TIG": lambda k: k.startswith("biggan"),
       "EP-DNN": lambda k: k.startswith("ep_dnn"),
       "Waterbirds": lambda k: k.startswith("waterbird")}
Ns = [100, 300, 500, 700, 1000]

def pooled(keys, m, metric, n):
    vals = []
    for k in keys:
        v = d[k][m][str(n)][metric]
        vals += v if isinstance(v, list) else [v]
    return np.mean(vals), np.std(vals)

# Square-root y-axis: most of what's worth reading (the "validated" line and EP-DNN,
# mostly single digits) lives near 0, while "total" ranges up to ~90. A square-root
# scale stretches the low end (0-5 takes ~22% of the height, 0-20 ~45%) while keeping a
# single, named, monotone mapping over the full 0-100 range.
Y_TICKS = [0, 5, 20, 50, 100]


def _y_forward(y):
    return np.sqrt(np.clip(y, 0, None))


def _y_inverse(p):
    return np.square(p)

# Bold everywhere (titles/axis labels/tick labels/legend) -- these rcParams cover
# set_title/set_ylabel/supxlabel and tick labels; legend text is bolded separately below
# via prop={"weight": "bold"} since rcParams has no legend.labelweight key.
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
fig, axes = plt.subplots(1, 5, figsize=(8.0, 2.3), sharex=True)
colors = {"total": "#1f4e79", "validated": "#c0504d"}
styles = {"idg": "-", "ig2": "--"}
summary = {}
for ax, (name, f) in zip(axes, fam.items()):
    keys = [k for k in d if f(k)]
    for metric in ["total", "validated"]:
        for m in ["idg", "ig2"]:
            mu, sd = zip(*[pooled(keys, m, metric, n) for n in Ns])
            mu, sd = np.array(mu), np.array(sd)
            summary[(name, m, metric)] = (mu, sd)
            c = colors[metric]
            ax.plot(Ns, mu, styles[m], color=c, lw=0.9)
            ax.plot(Ns[:-1], mu[:-1], "o", color=c, ms=3)
            ax.plot(Ns[-1], mu[-1], "o", color=c, ms=3, mfc="white")
            ax.fill_between(Ns[:-1], (mu-sd)[:-1], (mu+sd)[:-1], color=c, alpha=0.12, lw=0)
    ax.set_title(f"{name} ({len(keys)})", fontsize=8, fontweight="bold")
    ax.set_xticks(Ns)
    # Rotated (not horizontal) so 4-digit labels like "1000" don't need to fit side by
    # side with their neighbors in a ~1.3in-wide panel -- horizontal bold text at a
    # legible size was overlapping ("100"/"300" running into each other).
    ax.set_xticklabels([str(n) for n in Ns], rotation=45, ha="right", fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_yscale("function", functions=(_y_forward, _y_inverse))
    ax.set_ylim(0, 100)
    ax.set_yticks(Y_TICKS)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_ylabel("# features", fontweight="bold")
fig.supxlabel("Number of test inputs $N$", fontsize=8, fontweight="bold", y=0.16)
from matplotlib.lines import Line2D
h = [Line2D([],[],color=colors["total"],lw=1.5,label="$|L|$ (all features)"),
     Line2D([],[],color=colors["validated"],lw=1.5,label="$|V|$ (validated)"),
     Line2D([],[],color="k",ls="-",lw=0.9,label="IDG"),
     Line2D([],[],color="k",ls="--",lw=0.9,label="IG$^2$"),
     Line2D([],[],color="k",marker="o",mfc="white",ls="",ms=3,label="$N{=}1000$ (single draw)")]
fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04),
           prop={"weight": "bold", "size": 7})
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig("results/rq1.png", dpi=200, bbox_inches="tight", pad_inches=0)
for k,(mu,sd) in summary.items(): print(k, np.round(mu,1), np.round(sd,1))