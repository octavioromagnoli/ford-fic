#!/usr/bin/env python
"""Figuras y tablas del informe final (`docs/informe/`), desde salidas que ya existen.

    python scripts/make_report_figures.py --config configs/report_figures.yaml

No entrena, no puntúa y no abre el test: lee los CSV/JSON que dejó `scripts/eval_test.py` en
`experiments/test-*`, el de `scripts/explain_perm_seq.py` en `experiments/explain-gru-final/`, los YAML
de features, y las tablas auxiliares conservadas en `configs/report_figures.yaml`.
Escribe PDF en `output.figures_dir` y fragmentos `.tex` en `output.tables_dir`, que el informe incluye.

Lo que sale de `experiments/` se lee del archivo, nunca se tipea: si una reproducción de la GRU da otro
número (no es bit a bit, `docs/reproducibilidad.md`), la tabla del informe cambia sola.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_dir, load_config, resolve_path  # noqa: E402


# --------------------------------------------------------------------------------------------- estilo
def setup_style(style: dict) -> dict:
    c = style["colors"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": style["font_size"],
        "axes.edgecolor": c["grid"],
        "axes.linewidth": 0.6,
        "axes.labelcolor": c["text_secondary"],
        "axes.titlesize": style["font_size"] + 1,
        "axes.titleweight": "bold",
        "axes.titlecolor": c["text"],
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": c["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": c["text_secondary"],
        "ytick.color": c["text_secondary"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    return c


def pct(x: float, nd: int = 1) -> str:
    """0,3125 -> '31,2' (coma decimal, como el resto del informe)."""
    return f"{100 * x:.{nd}f}".replace(".", ",")


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def save(fig, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


# --------------------------------------------------------------------------------------------- test
def load_test(cfg: dict) -> pd.DataFrame:
    t = cfg["test"]
    curve = pd.read_csv(resolve_path(t["curve"]))
    curve = curve[curve["subset"] == t["subset"]].copy()
    curve["budget"] = curve["budget"].astype(str)
    curve["b"] = pd.to_numeric(curve["budget"], errors="coerce")  # NaN en la fila "primary"
    missing = set(t["models"]) - set(curve["model"])
    if missing:
        raise SystemExit(f"faltan modelos en {t['curve']}: {sorted(missing)}")
    return curve


def fig_test_curve(cfg: dict, c: dict, curve: pd.DataFrame, out: Path, width: float) -> None:
    """Detección vs presupuesto de falsas alarmas en test, con el nulo del finalista."""
    t = cfg["test"]
    num = curve[curve["b"].notna()].copy()
    num["budget"] = num["b"]
    fig, ax = plt.subplots(figsize=(width, 3.1))
    null = num[num["model"] == t["null_model"]].sort_values("budget")
    ax.fill_between(100 * null["budget"], 0, 100 * null["null_p95"], color=c["null_band"], lw=0,
                    label="Azar: hasta el p95 del nulo")
    ax.plot(100 * null["budget"], 100 * null["null_mean"], color=c["muted"], lw=1.2, ls="-",
            label="Azar: media del nulo")
    order = [m for m in t["models"] if t["models"][m]["role"] == "muted"] + \
            [m for m in t["models"] if t["models"][m]["role"] != "muted"]
    for m in order:
        spec = t["models"][m]
        d = num[num["model"] == m].sort_values("budget")
        color = c[spec["role"]]
        lw = 2.2 if spec["role"] == "focus" else (1.6 if spec["role"] != "muted" else 1.0)
        ax.plot(100 * d["budget"], 100 * d["detection"], color=color, lw=lw,
                marker="o" if spec["role"] != "muted" else None, ms=4, zorder=3 if spec["role"] != "muted" else 2)
        if spec["role"] != "muted":
            last = d.iloc[-1]
            ax.annotate(spec["label"], (100 * last["budget"], 100 * last["detection"]),
                        xytext=(6, spec.get("label_dy", 0)), textcoords="offset points", va="center",
                        color=c["text"], fontsize=8)
    ax.text(30.6, 100 * null["null_mean"].iloc[-1], "Azar (media)", va="center", color=c["text_secondary"], fontsize=8)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 80)
    ax.set_xticks([2, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Autos sanos con falsa alarma (%)")
    ax.set_ylabel("Autos que fallan, avisados antes (%)")
    ax.set_title("Detección por auto en test [test] · 103 autos, 32 fallados")
    muted = [spec.get("short", spec["label"]) for spec in t["models"].values() if spec["role"] == "muted"]
    ax.text(0.01, 0.97, "Gris fino: " + ", ".join(muted), transform=ax.transAxes,
            va="top", color=c["text_secondary"], fontsize=7.5)
    fig.subplots_adjust(right=0.78)
    save(fig, out, "curva_test")




def table_test(cfg: dict, curve: pd.DataFrame, out: Path) -> None:
    """Tabla completa de modelos en test (anexo), desde curve.csv + operating.csv."""
    t = cfg["test"]
    op = pd.read_csv(resolve_path(t["operating"]))
    budgets = t["budgets_table"]
    rows = []
    for m, spec in t["models"].items():
        d = curve[curve["model"] == m]
        dets = [pct(d.loc[np.isclose(d["b"], b), "detection"].item()) for b in t["budgets_table"]]
        p = d.loc[d["budget"] == "primary", "detection"]
        prim = pct(p.item()) if len(p) else "—"
        auc = t["auc_from_memo"].get(m)
        auc_s = f"{auc:.3f}".replace(".", ",") if auc is not None else "—"
        rows.append(f"{tex_escape(spec['label'])} & " + " & ".join(dets) + f" & \\textbf{{{prim}}} & {auc_s} & "
                    f"{t['dev_from_memo'].get(m, '—')} \\\\")
    null = curve[curve["model"] == t["null_model"]]
    null_cells = []
    for b in t["budgets_table"]:
        row = null[np.isclose(null["b"], b)].iloc[0]
        null_cells.append(f"{pct(row['null_mean'])} / {pct(row['null_p95'])}")
    rows.append(r"\midrule" + "\nAzar, media / p95 (nulo de la GRU) & " + " & ".join(null_cells) + r" & — & — & — \\")
    head = " & ".join(f"{int(round(100 * b))}\\,\\%" for b in t["budgets_table"])
    body = "\n".join(rows)
    tex = (
        "% Generado por scripts/make_report_figures.py desde " + t["curve"] + " (no editar a mano).\n"
        "\\begin{tabular}{@{}l" + "r" * len(budgets) + "rrl@{}}\n\\toprule\n"
        f"Modelo & {head} & Prim. & AUC auto & dev (5 · 10 · 20\\,\\%) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )
    (out / "test_modelos.tex").write_text(tex, encoding="utf-8")
    print(f"  {out / 'test_modelos.tex'}")

    # punto de operación con el umbral fijado en dev
    op_rows = []
    for m, spec in t["models"].items():
        d = op[op["model"] == m].sort_values("budget")
        if d.empty:
            continue
        det = " · ".join(f"{100 * v:.0f}" for v in d["detection"])
        fa = " · ".join(pct(v) for v in d["fa_realized"])
        op_rows.append(f"{tex_escape(spec['label'])} & {det} & {fa} \\\\")
    tex = (
        "% Generado por scripts/make_report_figures.py desde " + t["operating"] + " (no editar a mano).\n"
        "\\begin{tabular}{@{}lll@{}}\n\\toprule\n"
        "Modelo & Detección (\\%) al 5 · 10 · 20\\,\\% & Falsas alarmas reales (\\%) \\\\\n\\midrule\n"
        + "\n".join(op_rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (out / "test_operacion.tex").write_text(tex, encoding="utf-8")
    print(f"  {out / 'test_operacion.tex'}")


# --------------------------------------------------------------------------------------------- explicabilidad
def fig_permutation(cfg: dict, c: dict, out: Path, width: float) -> None:
    e = cfg["explain"]
    u = pd.read_csv(resolve_path(e["units"]))
    u = u.sort_values("detection_drop_pts")
    base = u["detection"].iloc[0]
    fig, ax = plt.subplots(figsize=(width, 3.4))
    y = np.arange(len(u))
    ax.barh(y, u["detection_drop_pts"], color=c["focus"], height=0.62)
    labels = [f"{e['labels'].get(r.unit, r.unit)}  ·  {e['families'].get(r.family, r.family)}" for r in u.itertuples()]
    for i, v in enumerate(u["detection_drop_pts"]):
        ax.text(v + 0.4, i, f"−{v:.1f}".replace(".", ","), va="center", fontsize=7.5, color=c["text"])
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 32)
    ax.set_xlabel(f"Puntos de detección que se pierden al permutar la señal (base {base:.1f} %)".replace(".", ","))
    ax.set_title("En qué se apoya la GRU [dev]")
    save(fig, out, "permutacion")


# --------------------------------------------------------------------------------------------- EDA
def fig_profile(cfg: dict, c: dict, out: Path, width: float) -> None:
    p = cfg["from_memo"]["profile"]
    x = np.arange(len(p["bins"]))
    fig, ax = plt.subplots(figsize=(width * 0.55, 2.6))
    ax.axhline(0.5, color=c["text_secondary"], lw=1)
    ax.text(0.9, 0.507, "0,5 = no distingue", fontsize=7, color=c["text_secondary"], va="bottom")
    for name, s in p["series"].items():
        col = c[s["role"]]
        ax.plot(x, s["values"], color=col, lw=2 if s["role"] != "muted" else 1.2, marker="o", ms=4)
        ax.annotate(name, (x[-1], s["values"][-1]), xytext=(6, s.get("label_dy", 0)), textcoords="offset points",
                    va="center", fontsize=7.5, color=c["text"])
    ax.set_xticks(x, p["bins"])
    ax.set_xlim(-0.2, len(x) - 0.6)
    ax.set_ylim(0.2, 0.85)
    ax.set_ylabel("P(fallado > sano del mercado)")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title("La huella crece hacia el evento [dev]", fontsize=9)
    save(fig, out, "perfil_evento")






def fig_score_toward_event(cfg: dict, c: dict, out: Path, width: float) -> None:
    s = cfg["from_memo"]["score_toward_event"]
    fig, ax = plt.subplots(figsize=(width * 0.5, 2.3))
    x = np.arange(len(s["bins"]))
    colors = [c["muted"]] + [c["focus"]] * (len(x) - 1)
    ax.bar(x, s["values"], color=colors, width=0.62)
    ax.axhline(0.5, color=c["text_secondary"], lw=1)
    for i, v in enumerate(s["values"]):
        ax.text(i, v + 0.015, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8, color=c["text"])
    ax.set_xticks(x, [b.replace(" km", "\nkm") for b in s["bins"]], fontsize=7.5)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Rango percentil medio")
    ax.set_title("El puntaje sube hacia el evento [test]")
    save(fig, out, "score_evento")


# --------------------------------------------------------------------------------------------- features
def table_features(cfg: dict, out: Path) -> None:
    """Diccionario de features en dos columnas (anexo): `nombre: agregador(columna derivada)` por familia."""
    spec = yaml.safe_load(resolve_path(cfg["features"]["spec"]).read_text(encoding="utf-8"))
    fam_names = {"A_thermal": "A · Térmica y trayectos cortos", "B_regeneration": "B · Regeneración (DPF)",
                 "C_usage": "C · Uso", "D_severity": "D · Severidad (mensajes, aceite, consumo)"}
    brk = lambda t: tex_escape(t).replace("\\_", "\\_\\allowbreak{}")  # noqa: E731
    lines, n_model, n_aux = [], 0, 0
    families = list(spec["families"].items()) + [("window", [
        {"name": "n_trips_window", "column": "*", "agg": "count"},
        {"name": "window_km_covered", "column": "OdometerTripEnd", "agg": "rango"}])]
    fam_names["window"] = "Control de ventana"
    for fam, items in families:
        lines.append(f"\\item[] \\textbf{{{fam_names.get(fam, fam)}}}")
        for it in items:
            aux = it.get("aux", False)
            n_aux += aux
            n_model += not aux
            prefix = "aux\\_" if aux else ""
            lines.append(f"\\item[] \\texttt{{{prefix}{brk(it['name'])}}}: {brk(it['agg'])}"
                         f"(\\texttt{{{brk(it['column'])}}})")
    tex = (
        "% Generado por scripts/make_report_figures.py desde " + cfg["features"]["spec"] + " (no editar a mano).\n"
        f"% {n_model} features de modelo + {n_aux} aux.\n"
        "\\begin{multicols}{2}\n\\raggedright\n\\begin{itemize}[leftmargin=0pt,itemsep=0pt,label={}]\n"
        + "\n".join(lines) + "\n\\end{itemize}\n\\end{multicols}\n"
    )
    (out / "features.tex").write_text(tex, encoding="utf-8")
    print(f"  {out / 'features.tex'}  ({n_model} de modelo, {n_aux} aux)")
    if n_model != 53:
        raise SystemExit(f"se esperaban 53 features de modelo, salieron {n_model}")

    seq = yaml.safe_load(resolve_path(cfg["features"]["sequence"]).read_text(encoding="utf-8"))
    sq = seq["sequence"]
    rows = []
    for ch in sq["channels"]:
        rows.append(f"\\texttt{{{tex_escape(ch['name'])}}} & {ch['source']} & \\texttt{{{tex_escape(ch['column'])}}} & "
                    f"{ch['agg']} & {ch['fill']}{(' · ' + ch['transform']) if ch.get('transform') else ''} \\\\")
    tex = (
        "% Generado por scripts/make_report_figures.py desde " + cfg["features"]["sequence"] + " (no editar a mano).\n"
        f"% ventana {sq['lookback_km']} km en bins de {sq['bin_km']} km = {sq['lookback_km'] // sq['bin_km']} tramos.\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\nCanal & Fuente & Columna & Por tramo & Relleno \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (out / "canales.tex").write_text(tex, encoding="utf-8")
    print(f"  {out / 'canales.tex'}  ({len(sq['channels'])} canales)")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    figs = ensure_dir(resolve_path(cfg["output"]["figures_dir"]))
    tabs = ensure_dir(resolve_path(cfg["output"]["tables_dir"]))
    c = setup_style(cfg["style"])
    w = cfg["style"]["width_in"]

    curve = load_test(cfg)
    fig_test_curve(cfg, c, curve, figs, w)
    table_test(cfg, curve, tabs)
    fig_permutation(cfg, c, figs, w)
    fig_profile(cfg, c, figs, w)
    fig_score_toward_event(cfg, c, figs, w)
    table_features(cfg, tabs)

    # el número que el informe cita en el texto tiene que ser el del archivo
    f10 = json.loads(resolve_path(cfg["test"]["report_f10"]).read_text(encoding="utf-8"))
    lead = f10["summary"]["test"]["candidate_median_lead_km_at_10"]
    print(f"  anticipación mediana al 10 % (test, GRU 42/1/2): {lead:,.0f} km")


if __name__ == "__main__":
    main()
