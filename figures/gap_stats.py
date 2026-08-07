#!/usr/bin/env python3
"""Derive the fig:gap efficiency-profile stats from measured logs into a CSV.

No number is hand-entered: everything below is read from the logs and computed.
Two sources are combined.

  1. WORK anchor (hardware-independent mask/model geometry) --
     logs/tile_skip_amplification/summary.json, workload block WORKLOAD.
     Gives the absolute per-run MAC counters (A, Ddc) and the scheduled/useful
     ratios (A/D, E/D). D_ours = Ddc / D_dc_over_D; E = (E/D) * D_ours.

  2. LATENCY (device-specific) -- the dynconv_pose run at LATENCY_RUN,
     per-backend mean/median/std of GPU-event latency (torch.cuda.Event, full
     model, per frame). No CPU/wall field exists in these logs.

Unit: the counters are MACs. DeltaCNN's n_active_flops (conv_kernel.cu) and
dynconv's flops_per_position (layers.py) both accumulate out*in*kh*kw with no
x2, i.e. multiply-accumulates. The paper reports FLOPS = 2 * MAC; MAC_TO_FLOP
makes that explicit.

Throughput (the required approximation): dense-model FLOPS scaled by the
fraction of dense work a path issues, divided by latency.
  issued work:  dense = D (all pixels), tile skipping = A (scheduled tiles),
                gather-scatter = E (exact active set).
  useful work:  E for every path (the mask's required minimum).
  raw   T     = issued_FLOP / latency
  eff   T_eff = useful_FLOP / latency = eta * T
  ideal floor = t_dense * (E/D): latency if useful work ran at dense throughput.

Edit CONFIG to retarget another device run / workload; the work anchor is
device-independent, so only LATENCY_RUN changes across platforms.
"""
import csv
import json
import os

# ------------------------------------------------------------------- CONFIG
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(REPO, "logs")
AMP_SUMMARY = os.path.join(LOGS, "tile_skip_amplification", "summary.json")
WORKLOAD = "dynconv_pose"
LATENCY_RUN = os.path.join(
    LOGS, "dynconv_pose", "agx-orin-30w-locked-energy-v4-full-s0125-r3", "summary.json"
)
# Accuracy is set by the mask, not the device, so it is read once from the run
# that evaluated it (the AGX latency run skips accuracy). PCKh on MPII.
ACCURACY_RUN = os.path.join(
    LOGS, "dynconv_pose", "3080-v23-full-s0125-r3-1800mhz", "summary.json"
)
DEVICE_LABEL = "Jetson AGX Orin (30 W)"
MAC_TO_FLOP = 2

# backend key in the logs -> (display name, issued-work quantity: D | A | E)
SYSTEMS = [
    ("dense", "Dense", "D"),
    ("tile_skip", "Tile skipping", "A"),
    ("dynconv", "Gather-scatter", "E"),
]
OUT_CSV = os.path.join(REPO, "RAL", "figures", "gap_stats.csv")


def load_work_anchor():
    """Absolute per-frame MAC work (D, A, E) and ratios from the amp summary."""
    amp = json.load(open(AMP_SUMMARY))
    w = amp["workloads"][WORKLOAD]
    frames = w["frames"]
    raw = w["raw_counters"]
    Ddc = raw["Ddc"]                       # DeltaCNN padded-grid dense MACs (run total)
    A_run = raw["A"]                       # tile_skip scheduled MACs (run total)
    D_dc_over_D = w["D_dc_over_D"]
    E_over_D = w["E_over_D"]
    A_over_D = w["A_over_D"]
    D_run = Ddc / D_dc_over_D               # real-pixel dense MACs (run total)
    E_run = E_over_D * D_run                # exact useful MACs (run total)
    # cross-check: A_run/D_run should reproduce the logged A_over_D
    assert abs(A_run / D_run - A_over_D) < 5e-3, (A_run / D_run, A_over_D)
    per = lambda run: run / frames
    return {
        "frames": frames,
        "eta_tile": w["eta"],
        "alpha_tile": w["alpha"],
        "E_over_D": E_over_D,
        "A_over_D": A_over_D,
        # per-frame MACs
        "D": per(D_run),
        "A": per(A_run),
        "E": per(E_run),
    }


def load_latency():
    """Per-backend GPU-event latency stats (ms) from the device run."""
    s = json.load(open(LATENCY_RUN))
    out = {}
    for backend, entry in s["backends"].items():
        lat = entry.get("latency") or {}
        out[backend] = {
            "mean": lat.get("mean"),
            "median": lat.get("median"),
            "std": lat.get("std"),
            "count": lat.get("count"),
        }
    return out, s["backends"]["dense"].get("tf32")


def load_accuracy():
    """Per-backend PCKh (%) from the run that evaluated accuracy."""
    s = json.load(open(ACCURACY_RUN))
    out = {}
    for backend, entry in s["backends"].items():
        acc = (entry or {}).get("accuracy") or {}
        out[backend] = acc.get("pckh")
    return out


def main():
    work = load_work_anchor()
    lat, tf32 = load_latency()
    acc = load_accuracy()
    D_gflop = work["D"] * MAC_TO_FLOP / 1e9          # dense FLOP/frame
    E_gflop = work["E"] * MAC_TO_FLOP / 1e9          # useful FLOP/frame
    issued_mac = {"D": work["D"], "A": work["A"], "E": work["E"]}

    t_dense = lat["dense"]["mean"]
    ideal_floor_ms = t_dense * work["E_over_D"]      # useful work at dense T

    rows = []
    for backend, name, qty in SYSTEMS:
        t = lat[backend]["mean"]
        issued_gflop = issued_mac[qty] * MAC_TO_FLOP / 1e9
        raw_tflops = issued_gflop / t                # GFLOP/ms == TFLOPS
        eff_tflops = E_gflop / t                     # useful/t == eta*T
        rows.append({
            "system": name,
            "backend": backend,
            "device": DEVICE_LABEL,
            "tf32": tf32,
            "frames": work["frames"],
            "latency_ms_mean": round(t, 3),
            "latency_ms_median": round(lat[backend]["median"], 3),
            "latency_ms_std": round(lat[backend]["std"], 3),
            "eta": round(work["eta_tile"], 4) if backend == "tile_skip"
                   else (1.0 if backend == "dynconv" else round(work["E_over_D"], 4)),
            "issued_gflop_per_frame": round(issued_gflop, 4),
            "useful_gflop_per_frame": round(E_gflop, 4),
            "dense_gflop_per_frame": round(D_gflop, 4),
            "raw_tflops": round(raw_tflops, 5),
            "eff_tflops": round(eff_tflops, 5),
            "pckh": round(acc[backend], 2) if acc.get(backend) is not None else "",
            "speedup_vs_dense": round(t_dense / t, 3),
            "ideal_floor_ms": round(ideal_floor_ms, 3),
            "gap_to_floor_x": round(t / ideal_floor_ms, 3),
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # human-readable echo so the numbers are auditable at generation time
    print(f"# work anchor : {AMP_SUMMARY}  ({WORKLOAD}, {work['frames']} frames)")
    print(f"# latency run : {LATENCY_RUN}  (tf32={tf32})")
    print(f"# dense FLOP/frame = {D_gflop:.3f} GFLOP ; useful = {E_gflop:.3f} GFLOP "
          f"(eta_tile={work['eta_tile']:.4f}, alpha={work['alpha_tile']:.3f})")
    print(f"# ideal sparsity floor = {ideal_floor_ms:.2f} ms (= t_dense*E/D)")
    hdr = ("system", "lat(ms)", "issued_GF", "raw_TF", "eff_TF", "spdup", "gap/floor")
    print("{:<15}{:>9}{:>11}{:>9}{:>9}{:>8}{:>10}".format(*hdr))
    for r in rows:
        print("{:<15}{:>9.2f}{:>11.3f}{:>9.4f}{:>9.4f}{:>8.2f}{:>10.2f}".format(
            r["system"], r["latency_ms_mean"], r["issued_gflop_per_frame"],
            r["raw_tflops"], r["eff_tflops"], r["speedup_vs_dense"],
            r["gap_to_floor_x"]))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
