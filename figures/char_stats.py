#!/usr/bin/env python3
"""Derive every number of the Characterization figures and table (RAL/src/characterization.tex) from the suite's
results in logs/wise2_char/<device>/ (experiments/wise2/char/, one script per insight).  Writes one CSV per figure
next to this script; the plot scripts (plot_char_*.py) read only these CSVs.

    python3 RAL/figures/char_stats.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs/wise2_char"
DEVICES = (("rtx_3080_sm86", "RTX 3080"), ("orin_sm87", "AGX Orin"))
DTYPES = ("fp32", "fp16")
REASONS = ("imc_miss", "barrier", "long_scoreboard", "short_scoreboard", "wait", "membar")


def load(dev, name):
    return json.loads((LOGS / dev / name).read_text())


def write(name, rows):
    path = HERE / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"{name}: {len(rows)} rows")


def short(label):
    """Layer label -> 'H² C→K r×r[s2]' (the input resolution of the layer)."""
    p = dict(zip(("n", "hw", "c", "k", "r", "st"), label.split("_")[:6]))
    h = p["hw"].split("x")[0]
    s = "s2" if p["st"] == "st2x2" else ""
    return f"{h}² {p['c'][1:]}→{p['k'][1:]} {p['r'][1:]}{s}"


def affine():
    rows = []
    for dev, dname in DEVICES:
        for p in DTYPES:
            s = load(dev, f"i1_summary_{p}.json")
            for i, a in enumerate(s["alphas"]):
                rows.append(dict(device=dname, dtype=p.upper(), alpha=a, best=s["curves"]["best"][i],
                                 tuned=s["curves"]["tuned"][i], single=s["curves"]["single"][i], fit_a=s["fit_a"],
                                 fit_b=s["fit_b"], r2_layer_median=s["r2_layer_median"],
                                 intercept_share_median=s["intercept_share_median"]))
    write("char_affine.csv", rows)


def reduction():
    rows = []
    for dev, dname in DEVICES:
        for p in DTYPES:
            for r in load(dev, f"i2_summary_{p}.json")["rows"]:
                rows.append(dict(device=dname, dtype=p.upper(), layer=r["layer"], short=short(r["label"]),
                                 variant=r["variant"], n_w=r["n_w"], tiles=r["tiles"],
                                 sectors_per_request=r["per_tile"]["sectors_per_request"],
                                 inst=r["vs_dense"]["inst"], dram_feature=r["vs_dense"]["dram_feature_bytes"],
                                 l2=r["vs_dense"]["l2_sectors"], cycles=r["vs_dense"]["cycles"]))
    write("char_reduction.csv", rows)
    marg = []
    for dev, dname in DEVICES:
        for p in DTYPES:
            for m in load(dev, f"i2_summary_{p}.json").get("marginal", []):
                marg.append(dict(device=dname, dtype=p.upper(), layer=m["layer"], short=short(m["label"]),
                                 tiles_min=m["tiles"][0], tiles_max=m["tiles"][-1],
                                 **{k: m[k] for k in ("inst", "cycles", "l2_sectors", "dram_bytes")},
                                 r2_min=min(min(m[k + "_r2"]) for k in ("inst", "cycles", "l2_sectors", "dram_bytes"))))
    write("char_reduction_marginal.csv", marg)


def order():
    rows, ctr = [], []
    for dev, dname in DEVICES:
        for p in DTYPES:
            t = load(dev, f"i3_time_{p}.json")["rows"]
            for o in ("sorted", "shuffled", "uniform"):
                x = np.array([r[o] / r["real"] for r in t])
                q = np.percentile(x, [5, 25, 50, 75, 95])
                rows.append(dict(device=dname, dtype=p.upper(), order=o, samples=len(x), p5=q[0], p25=q[1],
                                 p50=q[2], p75=q[3], p95=q[4], max=float(x.max()),
                                 total=sum(r[o] for r in t) / sum(r["real"] for r in t)))
            for c in load(dev, f"i3_summary_{p}.json").get("counters", []):
                for o in ("real", "sorted", "shuffled", "uniform"):
                    ctr.append(dict(device=dname, dtype=p.upper(), layer=c["layer"], order=o, dram=c[o]["dram"],
                                    l1_hit=c[o]["l1_hit"], l2=c[o]["l2"]))
    write("char_order.csv", rows)
    write("char_order_counters.csv", ctr)


def intercept():
    rows = []
    for dev, dname in DEVICES:
        for p in DTYPES:
            for r in load(dev, f"i4_intercept_summary_{p}.json"):
                st = r.get("stalls", {})
                row = dict(device=dname, dtype=p.upper(), layer=r["layer"], short=short(r["label"]), point=r["point"],
                           n_w=r["n_w"], tiles=r["tiles"], us=r["us"], intercept_us=r["intercept_us"],
                           intercept_share=r["intercept_share"])
                for k in REASONS:
                    row[k] = st.get(k, math.nan)
                row["other"] = 1 - sum(st.get(k, 0) for k in REASONS) if st else math.nan
                rows.append(row)
    write("char_intercept.csv", rows)


def construction():
    rows = []
    for dev, dname in DEVICES:
        for p in DTYPES:
            s = load(dev, f"i4_construction_summary_{p}.json")
            for i, a in enumerate(s["alphas"]):
                rows.append(dict(device=dname, dtype=p.upper(), alpha=a, share=s["share"][i],
                                 construction_median_us=s["construction_median"][i],
                                 construction_sum_us=s["construction_sum"][i], conv_sum_us=s["conv_sum"][i],
                                 per_launch_min=s["per_launch_min"], per_launch_max=s["per_launch_max"],
                                 real_share_median=float(np.median(s["real"]["share"])),
                                 real_corr=s["real"]["corr_alpha_share"]))
    write("char_construction.csv", rows)


def cost():
    gain, mech = [], []
    for dev, dname in DEVICES:
        for p in DTYPES:
            for r in load(dev, f"i5_gain_{p}.json")["rows"]:
                gain.append(dict(device=dname, dtype=p.upper(), layer=r["layer"], gain_us=r["gain_us"]))
        c = load(dev, "i5_cost_summary.json")
        # host readback is measured over eager launches; the graph-based mechanisms over the static graph: report it
        # over the static graph too (it cannot run inside one)
        for name, us in (("Host readback", c["host_readback_us"] + c["eager_over_graph_us"]),
                         ("Conditional node", c["conditional_us"]),
                         ("Updatable node", c["updatable_per_layer_us"])):
            mech.append(dict(device=dname, mechanism=name, us=us, per_switch_us=c["updatable_per_switch_us"]
                             if name == "Updatable node" else 0.0))
    write("char_gain.csv", gain)
    write("char_cost.csv", mech)


if __name__ == "__main__":
    affine()
    reduction()
    order()
    intercept()
    construction()
    cost()
