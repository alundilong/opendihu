#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate manuscript appendix tables from an existing grouped OpenDiHu study.

Expected layout:
  opendihu_repeated_denervation_study/
    study_metadata.json
    g-1/
      group_metadata_g-1.json
      protocol_A/{healthy,death_25,death_50,death_75}/...
      protocol_B/{healthy,death_25,death_50,death_75}/...
    g-2/
    g-3/

The script does not regenerate cases. It reads the audit files already produced
by generate_grouped_denervation_study_slurm.py and creates:
  * compact manuscript-ready Markdown appendix
  * complete 24-case Markdown appendix
  * compact CSV tables
  * one CSV per simulation case
  * one master long-format CSV (group x protocol x stage x MU)
  * A/B pairing audit based on SHA256 of distribution/firing-gate files

Example:
  python generate_appendix_parameter_tables.py \
      --study-root opendihu_repeated_denervation_study \
      --out-dir manuscript_appendix_tables
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STAGES = ["healthy", "death_25", "death_50", "death_75"]
STAGE_DISPLAY = {
    "healthy": "Healthy",
    "death_25": "25% MU loss",
    "death_50": "50% MU loss",
    "death_75": "75% MU loss",
}
STAGE_KILL = {"healthy": 0.0, "death_25": 0.25, "death_50": 0.50, "death_75": 0.75}
EXPECTED_ACTIVE = {"healthy": 20, "death_25": 15, "death_50": 10, "death_75": 5}
MU_CLASS = {
    "type_I_slow": "S (type-I-associated)",
    "type_IIa_fast_intermediate": "FR (type-IIa-associated)",
    "type_IIa": "FR (type-IIa-associated)",
    "type_IIx_fast": "FF (type-IIx-associated)",
    "type_IIx": "FF (type-IIx-associated)",
}

PATCH_PARAMS = {
    "Conductivity": ("Electrophysiology", "Effective conductivity", "mS/cm"),
    "end_time": ("Simulation", "Total simulation time", "ms"),
    "stimulation_frequency": ("Stimulation", "Firing-table sampling frequency", "ms^-1"),
    "stimulation_frequency_jitter": ("Stimulation", "Stimulation-frequency jitter", "-"),
    "dt_0D": ("Numerical", "CellML / ODE time step", "ms"),
    "dt_1D": ("Numerical", "1D diffusion time step", "ms"),
    "dt_splitting": ("Numerical", "Operator-splitting time step", "ms"),
    "dt_3D": ("Numerical", "3D coupling time step", "ms"),
    "output_timestep_fibers": ("Output", "Fiber output interval", "ms"),
    "output_timestep_3D_emg": ("Output", "3D EMG output interval", "ms"),
    "output_timestep_surface": ("Output", "Surface EMG output interval", "ms"),
    "sampling_stride_x": ("Volume conductor", "3D mesh sampling stride x", "-"),
    "sampling_stride_y": ("Volume conductor", "3D mesh sampling stride y", "-"),
    "sampling_stride_z": ("Volume conductor", "3D mesh sampling stride z", "-"),
    "hdemg_electrode_offset_xy": ("HD-sEMG array", "Electrode-array offset", "cm"),
    "hdemg_inter_electrode_distance_z": ("HD-sEMG array", "Electrode spacing, longitudinal", "cm"),
    "hdemg_inter_electrode_distance_xy": ("HD-sEMG array", "Electrode spacing, transverse", "cm"),
    "hdemg_n_electrodes_z": ("HD-sEMG array", "Electrode number, longitudinal", "-"),
    "hdemg_n_electrodes_xy": ("HD-sEMG array", "Electrode number, transverse", "-"),
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ffloat(v: Any, default=float("nan")) -> float:
    try:
        if v is None or str(v).strip() == "": return default
        return float(v)
    except Exception:
        return default


def fint(v: Any, default=None):
    try:
        if v is None or str(v).strip() == "": return default
        return int(float(v))
    except Exception:
        return default


def fbool(v: Any) -> bool:
    if isinstance(v, bool): return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def number(v: Any, nd=3):
    try:
        x = float(v)
        if not math.isfinite(x): return ""
        return round(x, nd)
    except Exception:
        return v if v is not None else ""


def fmt(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, bool): return "Yes" if v else "No"
    try:
        x = float(v)
        if not math.isfinite(x): return ""
        if abs(x - round(x)) < 1e-12: return str(int(round(x)))
        return f"{x:.5g}"
    except Exception:
        return str(v).replace("\n", " ")


def find_first(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists(): return p
    return None


def discover_groups(root: Path, requested: Optional[Sequence[int]]) -> Dict[int, Path]:
    wanted = None if not requested else {int(g) for g in requested}
    out = {}
    for p in sorted(root.glob("g-*")):
        m = re.fullmatch(r"g-(\d+)", p.name)
        if p.is_dir() and m:
            g = int(m.group(1))
            if wanted is None or g in wanted: out[g] = p
    if wanted is not None and wanted - set(out):
        raise FileNotFoundError(f"Missing group(s): {sorted(wanted - set(out))}")
    if not out: raise FileNotFoundError(f"No g-N folders found under {root}")
    return out


def casedir(gdir: Path, protocol: str, stage: str) -> Path:
    return gdir / f"protocol_{protocol}" / stage


def find_mu_json(d: Path) -> Optional[Path]:
    return find_first([d / "motor_units.json"] + sorted(d.glob("motor_units_*.json")))


def find_a_summary(d: Path) -> Optional[Path]:
    p = d / "motor_unit_summary.csv"
    if p.exists(): return p
    hits = [x for x in sorted(d.glob("*motor_unit_summary*.csv")) if "protocol_B" not in x.name]
    return hits[0] if hits else None


def find_b_summary(d: Path) -> Optional[Path]:
    p = d / "protocol_B_motor_unit_summary.csv"
    if p.exists(): return p
    hits = sorted(d.glob("*protocol_B_motor_unit_summary*.csv"))
    return hits[0] if hits else None


def find_gen_meta(d: Path) -> Optional[Path]:
    return find_first([d / "generation_metadata.json"] + sorted(d.glob("generation_metadata_*.json")))


def find_b_meta(d: Path) -> Optional[Path]:
    return find_first([d / "protocol_B_metadata.json"] + sorted(d.glob("protocol_B_metadata_*.json")))


def find_patch(d: Path) -> Optional[Path]:
    return find_first([d / "opendihu_variables_patch.py"] + sorted(d.glob("opendihu_variables_patch_*.py")))


def safe_eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        return node.value
    if isinstance(node, ast.UnaryOp):
        v = safe_eval(node.operand)
        if isinstance(node.op, ast.USub): return -v
        if isinstance(node.op, ast.UAdd): return +v
    if isinstance(node, ast.BinOp):
        a, b = safe_eval(node.left), safe_eval(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Pow): return a ** b
    raise ValueError


def parse_patch(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try: out[n.targets[0].id] = safe_eval(n.value)
            except Exception: pass
    return out


def rows_by_mu(path: Optional[Path], key="mu_id_0_based") -> Dict[int, Dict[str, Any]]:
    if path is None: return {}
    out = {}
    for r in read_csv(path):
        mu = fint(r.get(key, r.get("mu_no_0_based")))
        if mu is not None: out[mu] = r
    return out


def load_case(g: int, protocol: str, stage: str, gdir: Path):
    d = casedir(gdir, protocol, stage)
    if not d.exists(): raise FileNotFoundError(d)
    mj = find_mu_json(d)
    if mj is None: raise FileNotFoundError(f"No motor_units.json in {d}")
    payload = read_json(mj)
    mus = {fint(mu.get("mu_no_0_based"), i): mu for i, mu in enumerate(payload.get("motor_units", []))}

    ad = casedir(gdir, "A", stage)
    asum_path = find_a_summary(ad)
    asum = rows_by_mu(asum_path)
    bsum_path = find_b_summary(d) if protocol == "B" else None
    bsum = rows_by_mu(bsum_path)
    gm_path = find_gen_meta(ad)
    gm = read_json(gm_path) if gm_path else {}
    bm_path = find_b_meta(d) if protocol == "B" else None
    bm = read_json(bm_path) if bm_path else {}

    rows = []
    for mu_no in sorted(mus):
        mu, s, bs = mus[mu_no], asum.get(mu_no, {}), bsum.get(mu_no, {})
        killed = fbool(mu.get("killed_or_silent", s.get("killed_or_silent", False))) or not fbool(mu.get("active", True))
        radius = ffloat(mu.get("radius", s.get("radius_um")))
        initial = fint(s.get("initial_fiber_count", s.get("target_healthy_fiber_count", mu.get("target_fiber_count_healthy"))))
        final = fint(s.get("final_assigned_fiber_count"), initial)
        applied = ffloat(mu.get("stimulation_frequency", s.get("stimulation_frequency_hz", 0.0)), 0.0)
        baseline = applied if protocol == "A" else ffloat(mu.get("baseline_stimulation_frequency", bs.get("baseline_frequency_hz")))
        raw_type = mu.get("type_name", s.get("mu_type_name", ""))
        rank = fint(mu.get("recruitment_rank_0_based", s.get("recruitment_rank_0_based")))
        expansion = ffloat(s.get("expansion_ratio", bs.get("expansion_ratio", mu.get("protocol_B_expansion_ratio", 1.0))), 1.0)
        rows.append({
            "group": f"g-{g}", "group_number": g, "protocol": protocol, "stage": stage,
            "stage_display": STAGE_DISPLAY[stage], "kill_fraction": STAGE_KILL[stage],
            "MU_ID": mu_no + 1, "mu_id_0_based": mu_no,
            "status": "Killed/silent" if killed else "Active",
            "MU_class": MU_CLASS.get(raw_type, str(raw_type).replace("_", " ")),
            "raw_type_name": raw_type,
            "recruitment_rank_0_based": rank,
            "recruitment_order_1_based": None if rank is None else rank + 1,
            "healthy_target_fiber_count": fint(mu.get("target_fiber_count_healthy", s.get("target_healthy_fiber_count", initial))),
            "initial_fiber_count": initial,
            "final_assigned_fiber_count": final,
            "electrically_active_fiber_count": 0 if killed else final,
            "reinnervated_fibers_acquired": "" if killed or initial is None or final is None else max(0, final - initial),
            "expansion_ratio": expansion,
            "radius_um": radius,
            "Am_cm^-1": 20000.0 / radius if math.isfinite(radius) and radius > 0 else float("nan"),
            "Cm_uF_cm2": ffloat(mu.get("cm", s.get("cm"))),
            "activation_start_s": ffloat(mu.get("activation_start_time", s.get("activation_start_time_s"))),
            "baseline_frequency_Hz": baseline,
            "applied_frequency_Hz": applied,
            "protocol_B_global_factor": "" if protocol == "A" else ffloat(mu.get("protocol_B_global_factor", bs.get("global_factor", bm.get("global_factor")))),
            "protocol_B_local_factor": "" if protocol == "A" else ffloat(mu.get("protocol_B_local_factor", bs.get("local_factor"))),
            "protocol_B_frequency_scale": "" if protocol == "A" else ffloat(mu.get("protocol_B_frequency_scale", bs.get("frequency_scale"))),
            "group_seed": gm.get("group_seed", payload.get("group_seed", "")),
            "stage_seed": gm.get("stage_seed", ""), "motor_unit_seed": gm.get("motor_unit_seed", ""),
        })
    info = {
        "case_dir": str(d.resolve()), "n_mu": len(rows),
        "active_mus": sum(r["status"] == "Active" for r in rows),
        "killed_mus": sum(r["status"] != "Active" for r in rows),
        "active_fibers": sum(fint(r["electrically_active_fiber_count"], 0) or 0 for r in rows),
        "assigned_fibers": sum(fint(r["final_assigned_fiber_count"], 0) or 0 for r in rows),
    }
    return rows, info


def case_table(rows):
    out = []
    for r in rows:
        x = {
            "MU ID": r["MU_ID"], "Status": r["status"], "MU class": r["MU_class"],
            "Recruitment order": r["recruitment_order_1_based"],
            "Healthy target fibers": r["healthy_target_fiber_count"],
            "Assigned fibers": r["final_assigned_fiber_count"],
            "Active fibers": r["electrically_active_fiber_count"],
            "Expansion ratio": number(r["expansion_ratio"], 3),
            "Radius (um)": number(r["radius_um"], 3),
            "Am (cm^-1)": number(r["Am_cm^-1"], 2),
            "Cm (uF/cm2)": number(r["Cm_uF_cm2"], 3),
            "Activation start (s)": number(r["activation_start_s"], 3),
            "Firing frequency (Hz)": number(r["applied_frequency_Hz"], 2),
        }
        if r["protocol"] == "B":
            x.update({
                "Protocol A baseline (Hz)": number(r["baseline_frequency_Hz"], 2),
                "Global factor": number(r["protocol_B_global_factor"], 3),
                "Local factor": number(r["protocol_B_local_factor"], 3),
                "Frequency scale B/A": number(r["protocol_B_frequency_scale"], 3),
            })
        out.append(x)
    return out


def structural_table(g: int, cases):
    healthy = cases[(g, "A", "healthy")]
    out = []
    for h in healthy:
        mu = h["MU_ID"]
        r = {
            "MU ID": mu, "MU class": h["MU_class"], "Recruitment order": h["recruitment_order_1_based"],
            "Healthy target fibers": h["healthy_target_fiber_count"],
            "Radius (um)": number(h["radius_um"], 3), "Am (cm^-1)": number(h["Am_cm^-1"], 2),
            "Cm (uF/cm2)": number(h["Cm_uF_cm2"], 3),
        }
        for st, short in [("healthy", "Healthy"), ("death_25", "25%"), ("death_50", "50%"), ("death_75", "75%")]:
            q = next(x for x in cases[(g, "A", st)] if x["MU_ID"] == mu)
            r[f"{short} status"] = q["status"]
            r[f"{short} assigned fibers"] = q["final_assigned_fiber_count"]
            r[f"{short} active fibers"] = q["electrically_active_fiber_count"]
            r[f"{short} expansion"] = number(q["expansion_ratio"], 3)
        out.append(r)
    return out


def firing_table(g: int, cases):
    healthy = cases[(g, "A", "healthy")]
    out = []
    for h in healthy:
        mu = h["MU_ID"]
        r = {"MU ID": mu, "MU class": h["MU_class"], "Recruitment order": h["recruitment_order_1_based"]}
        for st, short in [("healthy", "Healthy"), ("death_25", "25%"), ("death_50", "50%"), ("death_75", "75%")]:
            a = next(x for x in cases[(g, "A", st)] if x["MU_ID"] == mu)
            b = next(x for x in cases[(g, "B", st)] if x["MU_ID"] == mu)
            r[f"{short} status"] = a["status"]
            r[f"A {short} (Hz)"] = number(a["applied_frequency_Hz"], 2)
            r[f"B {short} (Hz)"] = number(b["applied_frequency_Hz"], 2)
        out.append(r)
    return out


def study_design(root: Path, groups, all_rows):
    sm = read_json(root / "study_metadata.json") if (root / "study_metadata.json").exists() else {}
    g0 = min(groups)
    h = [r for r in all_rows if r["group_number"] == g0 and r["protocol"] == "A" and r["stage"] == "healthy"]
    fibers = sum(fint(r["final_assigned_fiber_count"], 0) or 0 for r in h)
    return [
        {"Parameter": "Independent stochastic realizations", "Value": len(groups), "Unit/definition": ", ".join(f"g-{g}" for g in groups)},
        {"Parameter": "Structural conditions", "Value": 4, "Unit/definition": "Healthy, 25%, 50%, 75% MU loss"},
        {"Parameter": "Stimulation protocols", "Value": 2, "Unit/definition": "A and B"},
        {"Parameter": "Principal simulations", "Value": len(groups) * 8, "Unit/definition": f"{len(groups)} x 4 x 2"},
        {"Parameter": "Motor units", "Value": len(h), "Unit/definition": "per simulation"},
        {"Parameter": "Computational fibers", "Value": fibers, "Unit/definition": "37 x 37" if fibers == 1369 else "fibers"},
        {"Parameter": "Simulation duration", "Value": sm.get("simulation_duration_s", ""), "Unit/definition": "s"},
        {"Parameter": "Statistical replicate", "Value": "stochastic realization", "Unit/definition": "group, not electrode"},
        {"Parameter": "Paired Protocol A/B design", "Value": sm.get("paired_protocol_design", True), "Unit/definition": "same structure within group/stage"},
    ]


def common_params(groups):
    g = min(groups); gd = groups[g]
    gp = find_first([gd / f"group_metadata_g-{g}.json", gd / "group_metadata.json"])
    gmeta = read_json(gp) if gp else {}
    mp = find_gen_meta(casedir(gd, "A", "healthy")); m = read_json(mp) if mp else {}
    s = m.get("scenario_parameters", {})
    rows = [
        {"Category": "MU population", "Parameter": "Number of MUs", "Value": m.get("n_motor_units", 20), "Unit/definition": "MUs"},
        {"Category": "Fiber grid", "Parameter": "Grid", "Value": f'{m.get("n_fibers_x", 37)} x {m.get("n_fibers_y", 37)}', "Unit/definition": "computational fibers"},
        {"Category": "MU composition", "Parameter": "Slow/type-I-associated fraction", "Value": m.get("type1_fraction", ""), "Unit/definition": "MU-pool fraction"},
        {"Category": "MU composition", "Parameter": "FR/type-IIa-associated fraction", "Value": m.get("type2a_fraction", ""), "Unit/definition": "MU-pool fraction"},
        {"Category": "MU composition", "Parameter": "FF/type-IIx-associated fraction", "Value": 1 - float(m.get("type1_fraction", .5)) - float(m.get("type2a_fraction", .35)), "Unit/definition": "MU-pool fraction"},
        {"Category": "MU size", "Parameter": "Exponential size basis b", "Value": m.get("mu_size_basis", gmeta.get("mu_size_basis", "")), "Unit/definition": "target fibers proportional to b^rank"},
        {"Category": "MU organization", "Parameter": "Spatial rank shuffle", "Value": m.get("spatial_rank_shuffle", ""), "Unit/definition": "MU ID independent of recruitment rank"},
        {"Category": "MU territory", "Parameter": "Territory sigma factor", "Value": m.get("territory_sigma_factor", ""), "Unit/definition": "-"},
        {"Category": "MU territory", "Parameter": "Territory size scaling", "Value": m.get("territory_size_scaling", ""), "Unit/definition": "-"},
        {"Category": "MU territory", "Parameter": "Neighbor repulsion factor", "Value": m.get("repulsion_factor", ""), "Unit/definition": "-"},
        {"Category": "Remodeling", "Parameter": "Scenario", "Value": m.get("scenario", gmeta.get("scenario", "")), "Unit/definition": "generator scenario"},
    ]
    names = {
        "p_death_slow": "Slow-MU vulnerability weight", "p_death_2a": "FR-MU vulnerability weight", "p_death_2x": "FF-MU vulnerability weight",
        "p_reinn_slow": "Slow-MU reinnervation competence", "p_reinn_2a": "FR-MU reinnervation competence", "p_reinn_2x": "FF-MU reinnervation competence",
        "sprout_radius": "Collateral sprouting radius", "sprout_sigma": "Collateral sprouting distance scale", "rescue_gain": "Collateral rescue gain",
    }
    for k, label in names.items():
        if k in s: rows.append({"Category": "Remodeling", "Parameter": label, "Value": s[k], "Unit/definition": "grid/model units" if "sprout" in k else "-"})
    return rows


def protocol_b_params(groups):
    g = min(groups); gd = groups[g]
    pth = find_first([gd / f"group_metadata_g-{g}.json", gd / "group_metadata.json"])
    gm = read_json(pth) if pth else {}; p = gm.get("protocol_B_parameters", {})
    if not p:
        q = find_b_meta(casedir(gd, "B", "death_50")); bm = read_json(q) if q else {}
        p = {"max_global_factor": bm.get("max_global_factor"), "global_gain": bm.get("global_gain"), "expansion_alpha": bm.get("expansion_alpha"), "max_frequency": bm.get("max_frequency"), "min_frequency": bm.get("min_frequency")}
    return [
        {"Parameter": "Protocol B rule", "Value": "f_B = clip(f_A * G * L_j, f_min, f_max)", "Unit/definition": "prescribed increased rate coding"},
        {"Parameter": "Global factor", "Value": "G = min(G_max, alpha_active^(-gamma))", "Unit/definition": "alpha_active = active MU fraction"},
        {"Parameter": "Local factor", "Value": "L_j = E_j^(-beta)", "Unit/definition": "E_j = remodeled/Healthy MU fiber-count ratio"},
        {"Parameter": "G_max", "Value": p.get("max_global_factor", ""), "Unit/definition": "-"},
        {"Parameter": "gamma", "Value": p.get("global_gain", ""), "Unit/definition": "-"},
        {"Parameter": "beta", "Value": p.get("expansion_alpha", ""), "Unit/definition": "-"},
        {"Parameter": "Minimum firing rate", "Value": p.get("min_frequency", ""), "Unit/definition": "Hz"},
        {"Parameter": "Maximum firing rate", "Value": p.get("max_frequency", ""), "Unit/definition": "Hz"},
    ]


def numerical_params(groups):
    g = min(groups); gd = groups[g]
    pp = find_patch(casedir(gd, "A", "healthy")); a = parse_patch(pp) if pp else {}
    rows = []
    for k, (cat, label, unit) in PATCH_PARAMS.items():
        if k in a: rows.append({"Category": cat, "Parameter": label, "Value": a[k], "Unit": unit, "Source variable": k})
    nx, nz = fint(a.get("hdemg_n_electrodes_xy")), fint(a.get("hdemg_n_electrodes_z"))
    if nx and nz: rows.append({"Category": "HD-sEMG array", "Parameter": "Total electrodes", "Value": nx * nz, "Unit": "electrodes", "Source variable": "derived"})
    mp = find_gen_meta(casedir(gd, "A", "healthy"))
    if mp:
        m = read_json(mp)
        rows += [
            {"Category": "Fiber model", "Parameter": "Fiber grid", "Value": f'{m.get("n_fibers_x", "")} x {m.get("n_fibers_y", "")}', "Unit": "computational fibers", "Source variable": "generation metadata"},
            {"Category": "Fiber model", "Parameter": "Total fibers", "Value": m.get("n_fibers_total", ""), "Unit": "fibers", "Source variable": "n_fibers_total"},
            {"Category": "MU model", "Parameter": "Motor units", "Value": m.get("n_motor_units", ""), "Unit": "MUs", "Source variable": "n_motor_units"},
        ]
    return rows


def inventory(groups, infos):
    out = []
    for g, gd in groups.items():
        for st in STAGES:
            ad, bd = casedir(gd, "A", st), casedir(gd, "B", st)
            da, db = ad / "MU_fibre_distribution_37x37_20.txt", bd / "MU_fibre_distribution_37x37_20.txt"
            fa, fb = ad / "MU_firing_times_always.txt", bd / "MU_firing_times_always.txt"
            dmatch = sha256(da) == sha256(db) if da.exists() and db.exists() else ""
            fmatch = sha256(fa) == sha256(fb) if fa.exists() and fb.exists() else ""
            for p in ("A", "B"):
                q = infos[(g, p, st)]
                out.append({"Group": f"g-{g}", "Protocol": p, "Stage": STAGE_DISPLAY[st], "Scenario ID": f"g-{g}_{p}_{st}", "Active MUs": q["active_mus"], "Killed/silent MUs": q["killed_mus"], "Electrically active fibers": q["active_fibers"], "Assigned fibers": q["assigned_fibers"], "A/B distribution SHA256 match": dmatch, "A/B firing-gate SHA256 match": fmatch, "Case directory": q["case_dir"]})
    return out


def md_table(rows):
    if not rows: return "_No data._\n"
    heads = list(rows[0].keys())
    lines = ["| " + " | ".join(heads) + " |", "| " + " | ".join("---" for _ in heads) + " |"]
    for r in rows:
        vals = [fmt(r.get(h, "")).replace("|", r"\|") for h in heads]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def validate(groups, cases):
    warnings = []
    for g in groups:
        for p in ("A", "B"):
            for st in STAGES:
                rows = cases[(g, p, st)]
                if len(rows) != 20: warnings.append(f"g-{g} {p} {st}: {len(rows)} MUs (expected 20)")
                n = sum(r["status"] == "Active" for r in rows)
                if n != EXPECTED_ACTIVE[st]: warnings.append(f"g-{g} {p} {st}: {n} active MUs (expected {EXPECTED_ACTIVE[st]})")
        for st in STAGES:
            A = {r["MU_ID"]: r for r in cases[(g, "A", st)]}; B = {r["MU_ID"]: r for r in cases[(g, "B", st)]}
            for mu in A:
                if mu not in B or A[mu]["status"] != B[mu]["status"] or A[mu]["final_assigned_fiber_count"] != B[mu]["final_assigned_fiber_count"]:
                    warnings.append(f"g-{g} {st} MU {mu}: A/B structural pairing differs")
    return warnings


def main():
    ap = argparse.ArgumentParser(description="Generate manuscript appendix tables from an existing grouped OpenDiHu study.")
    ap.add_argument("--study-root", default="opendihu_repeated_denervation_study")
    ap.add_argument("--out-dir", default="manuscript_appendix_tables")
    ap.add_argument("--groups", type=int, nargs="*", default=None)
    ap.add_argument("--strict", action="store_true", help="Fail if the expected 20/15/10/5 active-MU design is not recovered.")
    args = ap.parse_args()

    root = Path(args.study_root).expanduser().resolve(); out = Path(args.out_dir).expanduser().resolve()
    csvdir, casedir_out = out / "csv", out / "case_tables"
    csvdir.mkdir(parents=True, exist_ok=True); casedir_out.mkdir(parents=True, exist_ok=True)
    groups = discover_groups(root, args.groups)

    cases, infos, all_rows = {}, {}, []
    for g, gd in groups.items():
        print(f"Reading g-{g}")
        for p in ("A", "B"):
            for st in STAGES:
                rows, info = load_case(g, p, st, gd)
                cases[(g, p, st)] = rows; infos[(g, p, st)] = info; all_rows += rows
                write_csv(casedir_out / f"g-{g}_{p}_{st}.csv", case_table(rows))

    warnings = validate(groups, cases)
    if args.strict and warnings: raise RuntimeError("Validation failed:\n  " + "\n  ".join(warnings))

    A1, A2, PB, NUM, INV = study_design(root, groups, all_rows), common_params(groups), protocol_b_params(groups), numerical_params(groups), inventory(groups, infos)
    write_csv(csvdir / "A1_study_design.csv", A1); write_csv(csvdir / "A2_common_model_parameters.csv", A2)

    structural, firing = {}, {}; n = 3
    for g in groups:
        structural[g] = structural_table(g, cases); write_csv(csvdir / f"A{n}_structural_g-{g}.csv", structural[g]); n += 1
    for g in groups:
        firing[g] = firing_table(g, cases); write_csv(csvdir / f"A{n}_firing_rates_g-{g}.csv", firing[g]); n += 1
    write_csv(csvdir / f"A{n}_protocol_B_rate_coding.csv", PB)
    write_csv(csvdir / f"A{n+1}_numerical_hdemg_parameters.csv", NUM)
    write_csv(csvdir / f"A{n+2}_case_inventory.csv", INV)

    master_fields = ["group","group_number","protocol","stage","stage_display","kill_fraction","MU_ID","mu_id_0_based","status","MU_class","raw_type_name","recruitment_rank_0_based","recruitment_order_1_based","healthy_target_fiber_count","initial_fiber_count","final_assigned_fiber_count","electrically_active_fiber_count","reinnervated_fibers_acquired","expansion_ratio","radius_um","Am_cm^-1","Cm_uF_cm2","activation_start_s","baseline_frequency_Hz","applied_frequency_Hz","protocol_B_global_factor","protocol_B_local_factor","protocol_B_frequency_scale","group_seed","stage_seed","motor_unit_seed"]
    write_csv(csvdir / "all_cases_mu_parameters_long.csv", all_rows, master_fields)

    parts = ["# Appendix: Computational study parameters\n", "Tables below were generated directly from the archived study files. Group denotes the independent stochastic realization.\n", "## Table A1. Overall computational study design\n", md_table(A1), "\n## Table A2. Common motor-unit and remodeling parameters\n", md_table(A2)]
    tn = 3
    for g in groups:
        parts += [f"\n## Table A{tn}. Motor-unit structural remodeling for stochastic realization g-{g}\n", "Assigned-fiber count is structural ownership; active-fiber count is zero for killed/silent MUs.\n", md_table(structural[g])]; tn += 1
    for g in groups:
        parts += [f"\n## Table A{tn}. Prescribed firing rates for stochastic realization g-{g}\n", "Protocol A and B share the same structural realization within each stage.\n", md_table(firing[g])]; tn += 1
    parts += [f"\n## Table A{tn}. Protocol B increased-rate-coding parameters\n", md_table(PB), f"\n## Table A{tn+1}. Numerical and HD-sEMG parameters\n", md_table(NUM), f"\n## Table A{tn+2}. Case inventory and A/B pairing audit\n", md_table(INV)]
    (out / "appendix_compact.md").write_text("\n".join(parts), encoding="utf-8")

    full = ["# Appendix: Complete motor-unit parameter tables for all simulation cases\n"]
    sn = 1
    for g in groups:
        for p in ("A", "B"):
            for st in STAGES:
                full += [f"\n## Table S{sn}. g-{g}, Protocol {p}, {STAGE_DISPLAY[st]}\n", md_table(case_table(cases[(g,p,st)]))]; sn += 1
    (out / "appendix_full_case_tables.md").write_text("\n".join(full), encoding="utf-8")

    readme = ["Appendix table generation complete", f"Study root: {root}", f"Groups: {', '.join('g-'+str(g) for g in groups)}", f"Cases: {len(groups)*8}", f"Master MU rows: {len(all_rows)}", "", "Recommended manuscript appendix:", f"  {out/'appendix_compact.md'}", "", "Complete 24-case documentation:", f"  {out/'appendix_full_case_tables.md'}", f"  {casedir_out}", "", "Machine-readable master table:", f"  {csvdir/'all_cases_mu_parameters_long.csv'}", "", "Validation warnings:"]
    readme += ["  None."] if not warnings else ["  - " + w for w in warnings]
    readme += ["", "Notes:", "  Am is calculated as 20000/radius_um, matching the OpenDiHu callback.", "  activation_start_time is reported in seconds because the generated callback multiplies it by 1e3.", "  final_assigned_fiber_count can be non-zero for a killed MU; electrically_active_fiber_count is zero for killed/silent MUs."]
    (out / "README_appendix_tables.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("\nAppendix tables generated successfully")
    print(f"  Output: {out}")
    print(f"  Cases : {len(groups)*8}")
    print(f"  Rows  : {len(all_rows)}")
    print(f"  Compact appendix: {out/'appendix_compact.md'}")
    print(f"  Full appendix   : {out/'appendix_full_case_tables.md'}")
    print(f"  Master CSV      : {csvdir/'all_cases_mu_parameters_long.csv'}")
    if warnings:
        print("\nWarnings:")
        for w in warnings: print("  - " + w)
    else:
        print("\nValidation: no design inconsistencies detected.")

if __name__ == "__main__":
    main()
