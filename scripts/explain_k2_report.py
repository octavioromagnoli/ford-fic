"""Figuras y `cases.md` de la explicabilidad de K2, desde los archivos de `scripts/explain_k2.py`.

    python scripts/explain_k2.py --config configs/explain_k2.yaml --figures-only

Lee solo `experiments/explain-k2/` y los textos del YAML: no recalcula ningún SHAP. Tres figuras:

* `beeswarm.png`: la variante ganadora sobre todas las filas de dev, agrupada en accionables,
  contexto y síntomas; el color es el percentil del valor de la feature y la columna de la derecha
  dice si el signo observado coincide con el físico esperado.
* `stability.png`: Jaccard del top 3 y acuerdo de signo por variante, con los autos como puntos.
* `waterfall_<auto>.png`: de la base de la flota a la salida del auto, con las accionables una por
  una y el contexto y los síntomas en una barra cada uno.

Paleta: la de referencia de la guía de visualización del repo (azul `#2a78d6` / rojo `#e34948`,
validados como par; gris para lo que no es accionable, siempre con etiqueta).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import ensure_dir, load_config  # noqa: E402
from src.eval.explain import feature_label, format_value, waterfall_steps  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
UP, DOWN = "#e34948", "#2a78d6"          # sube el riesgo / lo baja (y valor alto / bajo)
VALUE_CMAP = LinearSegmentedColormap.from_list("valor", [DOWN, BASELINE, UP])
CLASS_TITLES = {"accionable": "Accionables", "contexto": "Contexto", "sintoma": "Síntomas"}
VARIANT_NAMES = {"V1": "V1 · TreeSHAP del hazard", "V2": "V2 · Permutation del score",
                 "V3": "V3 · V1 medio de 3 repeticiones", "V4": "V4 · V3 por familia"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    })


def _num(value: float, digits: int = 2, sign: bool = False) -> str:
    return f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",")


# --------------------------------------------------------------------------------------------- #
# beeswarm
# --------------------------------------------------------------------------------------------- #
def plot_beeswarm(rows: pd.DataFrame, classes: dict[str, str], plaus: pd.DataFrame, texts: dict, *,
                  n_per_class: dict[str, int], seed: int, title: str) -> plt.Figure:
    units = [c.removeprefix("phi__") for c in rows.columns if c.startswith("phi__")]
    importance = rows[[f"phi__{u}" for u in units]].abs().mean().set_axis(units)
    groups = []
    for klass in ("accionable", "contexto", "sintoma"):
        members = importance[[u for u in units if classes.get(u) == klass]].sort_values(ascending=False)
        groups.append((klass, list(members.index[: n_per_class[klass]])))
    n_rows = sum(len(m) for _, m in groups) + len(groups) - 1
    fig, ax = plt.subplots(figsize=(9.2, 0.34 * n_rows + 1.6))
    rng = np.random.default_rng(seed)
    y, ticks, labels = n_rows - 1, [], []
    lo, hi = np.inf, -np.inf
    for g, (klass, members) in enumerate(groups):
        for i, unit in enumerate(members):
            x = rows[f"phi__{unit}"].to_numpy(dtype=float)
            lo, hi = min(lo, np.nanmin(x)), max(hi, np.nanmax(x))
            counts, edges = np.histogram(x, bins=60)
            density = counts[np.clip(np.digitize(x, edges[1:-1]), 0, 59)] / counts.max()
            jitter = rng.uniform(-1, 1, len(x)) * 0.36 * density
            values = rows.get(f"val__{unit}")
            if values is not None and values.notna().sum() > 2:
                pct = values.rank(pct=True).to_numpy(dtype=float)
                known = np.isfinite(pct)
                ax.scatter(x[~known], y + jitter[~known], s=4, color=GRID, lw=0, zorder=2)
                ax.scatter(x[known], y + jitter[known], s=4, c=VALUE_CMAP(pct[known]), lw=0, alpha=0.8, zorder=3)
            else:
                ax.scatter(x, y + jitter, s=4, color=MUTED, lw=0, alpha=0.6, zorder=3)
            ticks.append(y)
            labels.append(feature_label(unit, texts))
            if klass == "accionable" and unit in plaus.index:
                row = plaus.loc[unit]
                sign = {1: "+", -1: "−", 0: "0"}
                if int(row["expected_sign"]) == 0:
                    note = "sin hipótesis"
                else:
                    note = (f"física {sign[int(row['expected_sign'])]} · obs. {sign[int(row['observed_sign'])]} "
                            + ("✓" if bool(row["aligned"]) else "✗"))
                ax.annotate(note, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK2)
            if i == 0:
                ax.annotate(CLASS_TITLES[klass], xy=(0.0, y + 0.55), xycoords=("axes fraction", "data"),
                            xytext=(-4, 0), textcoords="offset points", ha="right", va="bottom",
                            fontsize=9, fontweight="bold", color=INK)
            y -= 1
        y -= 1
    ax.axvline(0.0, color=BASELINE, lw=1.0, zorder=1)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Contribución al log-odds del hazard (promedio de los 6 tramos de H) · ← baja el riesgo · sube →")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=12)
    mappable = plt.cm.ScalarMappable(cmap=VALUE_CMAP, norm=plt.Normalize(0, 1))
    bar = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.19, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["bajo", "mediana", "alto"])
    bar.set_label("valor de la feature (percentil)", color=INK2)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------- #
# estabilidad
# --------------------------------------------------------------------------------------------- #
def plot_stability(ev: dict) -> plt.Figure:
    winner = ev["choice"]["winner"]
    names = ["V1", "V2", "V3", "V4"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, metric, title in ((axes[0], "jaccard", "Jaccard del top 3 entre réplicas"),
                              (axes[1], "sign_agreement", "Acuerdo de signo entre réplicas")):
        for i, name in enumerate(names):
            per = pd.DataFrame(ev["stability_by_vehicle"][name])[metric].to_numpy(dtype=float)
            mean = float(ev["variants"][name]["stability"][metric])
            y = len(names) - 1 - i
            color = DOWN if name == winner else BASELINE
            ax.barh(y, mean, height=0.42, color=color, zorder=2)
            ax.scatter(per, y + rng.uniform(-0.13, 0.13, len(per)), s=10, color=INK2, alpha=0.45, lw=0, zorder=3)
            ax.annotate(_num(mean), xy=(mean, y), xytext=(0, 12), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    labels = []
    for name in names:
        res = ev["variants"][name]
        tag = "no elegible (top 1 de familia)" if name not in ev["choice"]["eligible"] else (
            "pasa la fidelidad" if res["passes_fidelity"] else "no pasa la fidelidad")
        labels.append(f"{VARIANT_NAMES[name]}\n{tag}" + (" · elegida" if name == winner else ""))
    axes[0].set_yticks(range(len(names) - 1, -1, -1), labels)
    n = ev["variants"]["V1"]["stability"]["n_vehicles"]
    fig.text(0.01, 0.01, f"Barras: media sobre los {n} autos que alertan con V al 5% en alguna repetición (puntos: cada auto). "
             "Réplicas: V1 y V2, las 3 repeticiones; V3 y V4, 3 tripletes disjuntos de repeticiones.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# --------------------------------------------------------------------------------------------- #
# waterfall
# --------------------------------------------------------------------------------------------- #
def plot_waterfall(steps: pd.DataFrame, base: float, output: float, *, title: str, subtitle: str) -> plt.Figure:
    n = len(steps) + 2
    fig, ax = plt.subplots(figsize=(9.2, 0.36 * n + 1.5))
    xmin = min(base, output, *(base + steps["value"].cumsum()))
    xmax = max(base, output, *(base + steps["value"].cumsum()))
    pad = 0.22 * (xmax - xmin + 1e-9)
    ax.set_xlim(xmin - pad, xmax + pad)

    def end_label(x: float, y: float, text: str, color: str) -> None:
        # A la derecha del punto, salvo que esté en el último tercio: ahí va a la izquierda.
        right = x < xmin + 0.66 * (xmax - xmin)
        ax.annotate(text, xy=(x, y), xytext=(8 if right else -8, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=8.5, color=color)

    y = n - 1
    ax.scatter([base], [y], s=36, color=INK2, zorder=3)
    end_label(base, y, f"base {_num(base)}", INK2)
    labels, ticks = ["base (hazard medio del entrenamiento)"], [y]
    running = base
    for _, s in steps.iterrows():
        y -= 1
        start, end = running, running + s["value"]
        color = (UP if s["value"] > 0 else DOWN) if s["kind"] == "accionable" else BASELINE
        ax.barh(y, end - start, left=start, height=0.5, color=color, zorder=2)
        ax.plot([end, end], [y - 0.25, y - 0.75], color=MUTED, lw=0.8, zorder=1)
        ax.annotate(_num(s["value"], sign=True), xy=(max(start, end), y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
        labels.append(("✓ " if s["in_message"] else "") + s["step"])
        ticks.append(y)
        running = end
    y -= 1
    ax.scatter([output], [y], s=36, color=INK, zorder=3)
    end_label(output, y, f"salida {_num(output)}", INK)
    labels.append("salida del auto (log-odds medio del hazard)")
    ticks.append(y)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("log-odds del hazard, promedio de los 6 tramos de H\n"
                  "rojo: la accionable sube el riesgo · azul: lo baja · gris: no es accionable")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=18)
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------- #
# cases.md
# --------------------------------------------------------------------------------------------- #
def _case_rows(ev: dict, vehicle: pd.DataFrame, messages: pd.DataFrame, texts: dict) -> list[dict]:
    label, repeat = ev["cases_label"], ev["cases_repeat"]
    winner = ev["choice"]["winner"] or "V1"
    actionables = ev["classification"]["actionable"]
    out = []
    for case in ev["cases"]:
        vid = case["vehicle_id"]
        ctx = vehicle.loc[(vehicle["variant"] == "V1") & (vehicle["label"] == label) & (vehicle["replicate"] == repeat)
                          & (vehicle["vehicle_id"] == vid)].iloc[0]
        replicate = 0 if winner in ("V3", "V4") else repeat
        vec = vehicle.loc[(vehicle["variant"] == winner) & (vehicle["label"] == label) & (vehicle["replicate"] == replicate)
                          & (vehicle["vehicle_id"] == vid)].iloc[0]
        phi = pd.Series({f: vec[f"phi__{f}"] for f in actionables}).sort_values(ascending=False)
        msg = messages.loc[(messages["vehicle_id"] == vid) & (messages["label"] == label) & (messages["repeat"] == repeat)]
        shown = {f["feature"] for f in json.loads(msg["factors"].iloc[0])} if len(msg) else set()
        top = []
        for f in phi.index[:3]:
            fmt = ((texts.get("features") or {}).get(f) or {}).get("format", "")
            val = format_value(ctx[f"val__{f}"], fmt) if fmt else _num(ctx[f"val__{f}"])
            ref = format_value(ctx[f"ref__{f}"], fmt) if fmt else _num(ctx[f"ref__{f}"])
            top.append(f"{feature_label(f, texts)} {_num(phi[f], sign=True)} ({val} vs {ref}){' ✓' if f in shown else ''}")
        out.append({"vehicle_id": vid, "rule": case["rule"], "max_score": float(ctx["max_score"]), "score": float(ctx["score"]),
                    "threshold": float(ctx["threshold"]), "alerted": bool(ctx["alerted"]), "failed": bool(ctx["failed"]),
                    "risk_level": ctx["risk_level"], "top": top,
                    "message": msg["message"].iloc[0] if len(msg) else "(sin mensaje: ninguna variante pasó la fidelidad)"})
    return out


def write_cases(ev: dict, rows: list[dict], path: Path, figures_dir: str) -> None:
    winner = ev["choice"]["winner"]
    lines = [
        "# Casos de la explicabilidad de K2",
        "",
        f"Generado por `scripts/explain_k2.py` ({ev['created_at']}). Solo dev; el test no se toca.",
        f"Variante elegida: **{winner}** ({VARIANT_NAMES.get(winner, winner)}). Casos elegidos por la regla del YAML: "
        f"repetición {ev['cases_repeat']}, etiqueta {'V' if ev['cases_label'] == 'corrected' else 'D'}, 5% de falsas alarmas; "
        "el score máximo más alto de cada categoría y, para el sano de riesgo bajo, el más bajo.",
        "",
        "Las contribuciones están en log-odds del hazard (promedio de los 6 tramos). ✓ = el factor llega al mensaje "
        "(pasa todos los filtros del preregistro). Valor del auto vs mediana de los sanos del mismo mercado × mes (train del fold).",
        "",
        "| auto | categoría | score máx. | score de los cortes explicados | umbral | ¿alertó? | ¿falló? | top 3 accionables (valor vs mediana sana) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['vehicle_id']}` | {r['rule']} | {_num(r['max_score'], 3)} | {_num(r['score'], 3)} | "
                     f"{_num(r['threshold'], 3)} | {'sí' if r['alerted'] else 'no'} | {'sí' if r['failed'] else 'no'} | "
                     + "<br>".join(r["top"]) + " |")
    lines.append("")
    for r in rows:
        lines += [f"## `{r['vehicle_id']}` · {r['rule']}", "",
                  f"![waterfall]({figures_dir}/waterfall_{r['vehicle_id']}.png)", "", "Mensaje al cliente, tal cual sale:", "",
                  "```text", r["message"], "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------------- #
def write_report(cfg: dict, out_dir: Path) -> None:
    """Figuras y cases.md desde los archivos del script (no recalcula SHAP)."""
    _style()
    ev = json.loads((out_dir / cfg["outputs"]["eval"]).read_text(encoding="utf-8"))
    ev["cases_label"], ev["cases_repeat"] = cfg["cases"]["label"], int(cfg["cases"]["repeat"])
    rows = pd.read_parquet(out_dir / cfg["outputs"]["shap_rows"])
    vehicle = pd.read_parquet(out_dir / cfg["outputs"]["shap_vehicle"])
    messages = pd.read_parquet(out_dir / "messages.parquet")
    texts = load_config(cfg["message"]["texts"])
    figures = ensure_dir(out_dir / cfg["outputs"]["figures_dir"])
    winner = ev["choice"]["winner"]
    cls = ev["classification"]
    classes = {**{u: "accionable" for u in cls["actionable"]}, **{u: "sintoma" for u in cls["symptom"]},
               **{u: "contexto" for u in cls["context"]}}

    if winner in ("V1", "V3"):
        source = rows.loc[rows["variant"].eq("V3")] if winner == "V3" else rows.loc[rows["variant"].eq("V1") & rows["set"].eq("official")]
        if winner == "V1":
            vals = rows.loc[rows["variant"].eq("V3"), ["row"] + [c for c in rows.columns if c.startswith("val__")]]
            source = source.drop(columns=[c for c in source.columns if c.startswith("val__")]).merge(vals, on="row")
    else:
        source = rows.loc[rows["variant"].eq(winner)]
    source = source.dropna(axis=1, how="all")
    plaus = pd.DataFrame(ev["plausibility"].get(winner, ev["plausibility"]["V3"])).set_index("feature")
    fig = plot_beeswarm(source, classes, plaus, texts, n_per_class={"accionable": 12, "contexto": 5, "sintoma": 5},
                        seed=int(cfg["seed"]),
                        title=f"Qué mira K2 ({winner}, {format_value(len(source), 'int')} cortes de dev fuera de fold)")
    fig.savefig(figures / "beeswarm.png", dpi=160)
    plt.close(fig)

    fig = plot_stability(ev)
    fig.savefig(figures / "stability.png", dpi=160)
    plt.close(fig)

    case_rows = _case_rows(ev, vehicle, messages, texts)
    allowed = set(ev.get("allowed_in_text", []))
    for case in case_rows:
        vid = case["vehicle_id"]
        replicate = 0 if winner in ("V3", "V4") else ev["cases_repeat"]
        vec_row = vehicle.loc[(vehicle["variant"] == (winner or "V1")) & (vehicle["label"] == ev["cases_label"])
                              & (vehicle["replicate"] == replicate) & (vehicle["vehicle_id"] == vid)].iloc[0]
        ctx = vehicle.loc[(vehicle["variant"] == "V1") & (vehicle["label"] == ev["cases_label"])
                          & (vehicle["replicate"] == ev["cases_repeat"]) & (vehicle["vehicle_id"] == vid)].iloc[0]
        units = [c.removeprefix("phi__") for c in vehicle.columns if c.startswith("phi__") and np.isfinite(vec_row[c])]
        vector = pd.Series({u: vec_row[f"phi__{u}"] for u in units})
        msg = messages.loc[(messages["vehicle_id"] == vid) & (messages["label"] == ev["cases_label"])
                           & (messages["repeat"] == ev["cases_repeat"])]
        shown = {f["feature"] for f in json.loads(msg["factors"].iloc[0])} if len(msg) else set()
        vals = pd.Series({u: ctx.get(f"val__{u}", np.nan) for u in units})
        refs = pd.Series({u: ctx.get(f"ref__{u}", np.nan) for u in units})
        steps = waterfall_steps(vector, classes, top=6, allowed_in_message=shown & allowed, texts=texts,
                                values=vals, references=refs)
        fig = plot_waterfall(steps, float(vec_row["base"]), float(vec_row["output"]),
                             title=f"{vid} · {case['rule'].split(' (')[0]} · score {_num(case['score'], 3)} (umbral {_num(case['threshold'], 3)})",
                             subtitle=f"{VARIANT_NAMES.get(winner, winner)} · valores: el auto vs la mediana sana comparable · ✓ llega al mensaje")
        fig.savefig(figures / f"waterfall_{vid}.png", dpi=160)
        plt.close(fig)

    write_cases(ev, case_rows, out_dir / cfg["outputs"]["cases"], cfg["outputs"]["figures_dir"])
