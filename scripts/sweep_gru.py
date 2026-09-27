#!/usr/bin/env python
"""Sweep bayesiano (Optuna TPE) de los hiperparámetros de la GRU, solo sobre dev.

    python scripts/sweep_gru.py --config configs/sweep_gru.yaml                  # lanza el sweep
    python scripts/sweep_gru.py --config configs/sweep_gru.yaml --summary        # resume uno que ya corrió
    python scripts/sweep_gru.py --config configs/sweep_gru.yaml --write-configs  # YAMLs de confirmación

**Qué barre.** Los hiperparámetros de `gru_seq` declarados en `space:` del YAML. El panel, los
folds (`splits_r3.json`), el holdout y la evaluación salen de `base_config`, así que cada trial
es la misma corrida que `train.py` haría con otros `model.params`: el recorte a dev lo hace
`select_dev()` y los folds los da `iter_repeats()` (regla 2).

**Qué optimiza.** La detección por auto con umbral exacto de `report_v2_models.py` (un auto
alerta si tiene `k` cortes seguidos sobre τ, y τ deja ≤ b de los sanos alertados), promediada
sobre los presupuestos `objective.budgets` y después sobre las R repeticiones. Cada repetición
se mide con sus propios scores: promediar scores entre repeticiones es un ensamble encubierto
(`run_cv`). Van al lado, sin optimizarse, el PR-AUC por fila y el AUC por auto dentro de
mercado × motor, que es donde se lee si el modelo ordena autos comparables (CLAUDE.md).

**Cómo corre.** El proceso principal crea el estudio en un `JournalFileStorage` (un archivo, sin
servidor) y lanza `n_workers` procesos con `threads_per_worker` hilos de torch cada uno. Cada
worker tiene su semilla del sampler (si no, los trials aleatorios del arranque se repiten). La
poda es por repetición: un trial cuyo promedio después de la rep 0 queda bajo la mediana de los
anteriores en ese mismo paso no sigue. Los workers dejan de empezar trials cuando no llegan a
terminar uno antes del límite, así que el sweep dura `timeout_minutes` y no un trial de más.

**Qué no dice.** El mejor trial es el máximo de muchos intentos contra los mismos autos y los
mismos folds: su número está inflado. `--write-configs` escribe las `top_k` con semillas del
modelo que el sweep no usó (`confirm_seeds`), para re-medirlas con `train.py` y compararlas con
`report_v2_models.py` (bootstrap pareado) contra la configuración de hoy.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_detection_null import vehicle_levels  # noqa: E402
from scripts.report_v2_models import detection_at, pairs_auc  # noqa: E402
from src.config import ensure_dir, load_config, repo_root, resolve_path  # noqa: E402

logger = logging.getLogger("sweep_gru")
KEY = ["vehicle_id", "cut_odo"]


# --- estudio ------------------------------------------------------------------------------

def out_dir(cfg: dict) -> Path:
    return ensure_dir(Path(resolve_path(cfg.get("output_dir", "experiments"))) / cfg["name"])


def open_study(cfg: dict, *, worker: int = 0):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # `group` del TPE
    study_cfg = cfg["study"]
    path =str(resolve_path(study_cfg["storage"]))
    ensure_dir(Path(path).parent)
    # El lock por symlink necesita privilegios en Windows; el de apertura exclusiva no.
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    s = study_cfg["sampler"]
    sampler = optuna.samplers.TPESampler(
        seed=int(s["seed"]) + worker,
        n_startup_trials=int(s["n_startup_trials"]),
        multivariate=bool(s.get("multivariate", True)),
        group=bool(s.get("group", False)),
        constant_liar=True,
    )
    p = study_cfg["pruner"]
    if p["name"] != "median":
        raise ValueError(f"pruner {p['name']!r}: solo está implementado `median`")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=int(p["n_startup_trials"]),
                                         n_warmup_steps=int(p["n_warmup_steps"]))
    return optuna.create_study(study_name=cfg["name"], storage=storage, sampler=sampler, pruner=pruner,
                               direction="maximize", load_if_exists=True)


def base_params(cfg: dict) -> dict[str, Any]:
    base = load_config(cfg["base_config"])
    if base["model"]["name"] != "gru_seq":
        raise ValueError(f"base_config entrena `{base['model']['name']}`, no `gru_seq`")
    return dict(base["model"]["params"])


def enqueue_base(study, cfg: dict) -> None:
    """La configuración de hoy como primer trial: la vara, medida con la misma cuenta."""
    params = base_params(cfg)
    fixed = {}
    for name, spec in cfg["space"].items():
        cond = spec.get("when")
        if cond and any(params.get(k) != v for k, v in cond.items()):
            continue
        value = params[name]
        fixed[name] = "none" if value is None else value
    study.enqueue_trial(fixed, user_attrs={"is_base": True}, skip_if_exists=True)


def sample(trial, space: dict) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        cond = spec.get("when")
        if cond and any(params.get(k) != v for k, v in cond.items()):
            continue
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                             step=int(spec.get("step", 1)), log=bool(spec.get("log", False)))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                               log=bool(spec.get("log", False)))
        else:
            raise ValueError(f"`{name}`: type {kind!r} (int | float | categorical)")
    return params


def to_model_params(base: dict, params: dict) -> dict:
    merged = {**base, **params}
    if merged.get("class_weight") == "none":
        merged["class_weight"] = None
    return merged


# --- worker -------------------------------------------------------------------------------

def init_trial_wandb(cfg: dict, name: str, config: dict):
    w = cfg.get("wandb", {})
    if not w.get("enabled", True):
        return None
    try:
        import wandb

        return wandb.init(project=w.get("project", "ford-fic"),
                          entity=os.environ.get("WANDB_ENTITY") or w.get("entity"),
                          name=name, group=w.get("group"), job_type=w.get("job_type", "sweep"),
                          tags=w.get("tags"), mode=os.environ.get("WANDB_MODE") or w.get("mode", "online"),
                          config=config, reinit="finish_previous")
    except Exception as exc:  # noqa: BLE001 - wandb caído no frena el sweep: todo queda en el journal
        logger.warning("wandb no inicializó (%s): el trial queda solo en el journal", exc)
        return None


def run_worker(cfg: dict, worker: int, deadline: float) -> None:
    import optuna
    import torch

    from scripts.train import load_or_make_splits, select_dev
    from src.training.cv import run_cv

    torch.set_num_threads(int(cfg["threads_per_worker"]))
    logging.getLogger("src").setLevel(logging.WARNING)
    logging.getLogger("scripts.train").setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_cfg = load_config(cfg["base_config"])
    panel = pd.read_parquet(resolve_path(base_cfg["data"]["panel"]))
    panel = select_dev(panel, base_cfg)
    splits = load_or_make_splits(panel, base_cfg)
    splits_cfg = base_cfg.get("splits", {})
    strict = bool(splits_cfg.get("strict", True))
    min_pos = splits_cfg.get("min_valid_positives")
    repeats = splits["repeats"]

    # mercado × motor por vehículo, para el AUC dentro de celda
    per_vehicle = panel.groupby("vehicle_id")[["static_SalesCountry_cd", "static_Engine"]].first()
    cell_of = per_vehicle["static_SalesCountry_cd"].astype(str) + "×" + per_vehicle["static_Engine"].astype(str)

    base = base_params(cfg)
    budgets = [float(b) for b in cfg["objective"]["budgets"]]
    k = int(cfg["objective"]["k_consecutive"])

    def objective(trial) -> float:
        params = sample(trial, cfg["space"])
        model_params = to_model_params(base, params)
        is_base = bool(trial.user_attrs.get("is_base", False))
        run = init_trial_wandb(cfg, f"{cfg['name']}-t{trial.number:03d}",
                               {**model_params, "trial": trial.number, "is_base": is_base,
                                "threads": int(cfg["threads_per_worker"]), "worker": worker})
        started = time.time()
        per_rep: list[float] = []
        try:
            for r, entry in enumerate(repeats):
                one = {**splits, "n_repeats": 1, "repeats": [entry]}
                pred, _ = run_cv(panel, one, model_name="gru_seq", model_params=model_params,
                                 strict_splits=strict, min_valid_positives=min_pos)
                pred = pred.sort_values(KEY).reset_index(drop=True)
                s = pred["score"].to_numpy(dtype=float)
                levels, event, _ = vehicle_levels(pred, s, k)
                dets = [detection_at(levels, event, b)[0] for b in budgets]
                ids = pred["vehicle_id"].to_numpy()
                first = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
                vmean = pd.Series(s).groupby(ids).mean().reindex(ids[first]).to_numpy()
                cells = cell_of.reindex(ids[first]).to_numpy()
                row = {
                    "detection": float(np.mean(dets)),
                    **{f"det_{int(round(100 * b))}": d for b, d in zip(budgets, dets)},
                    "pr_auc": float(average_precision_score(pred["label"], s)),
                    "auc_vehicle": pairs_auc(vmean, event, None),
                    "auc_within_cell": pairs_auc(vmean, event, cells),
                }
                for key, value in row.items():
                    trial.set_user_attr(f"r{r}_{key}", value)
                per_rep.append(row["detection"])
                running = float(np.mean(per_rep))
                if run is not None:
                    run.log({"repeat": r, "objective_running": running, **{f"rep/{k2}": v for k2, v in row.items()}})
                if r < len(repeats) - 1:
                    trial.report(running, step=r)
                    if trial.should_prune():
                        trial.set_user_attr("minutes", (time.time() - started) / 60)
                        if run is not None:
                            run.summary.update({"state": "pruned", "objective_running": running})
                        raise optuna.TrialPruned()
            value = float(np.mean(per_rep))
            minutes = (time.time() - started) / 60
            trial.set_user_attr("minutes", minutes)
            for key in ("detection", *[f"det_{int(round(100 * b))}" for b in budgets], "pr_auc",
                        "auc_vehicle", "auc_within_cell"):
                vals = [trial.user_attrs[f"r{r}_{key}"] for r in range(len(repeats))]
                trial.set_user_attr(f"mean_{key}", float(np.mean(vals)))
                trial.set_user_attr(f"sd_{key}", float(np.std(vals)))
            if run is not None:
                run.summary.update({"state": "complete", "objective": value, "minutes": minutes,
                                    **{k2: v for k2, v in trial.user_attrs.items() if k2.startswith(("mean_", "sd_"))}})
            logger.info("trial %d | %.4f | %.1f min | %s", trial.number, value, minutes, params)
            return value
        finally:
            if run is not None:
                run.finish()

    def stop_before_deadline(study, _trial) -> None:
        """No empezar un trial que no llega a terminar antes del límite."""
        done = [t.user_attrs.get("minutes") for t in study.get_trials(deepcopy=False)
                if t.state.is_finished() and t.user_attrs.get("minutes") and "mean_detection" in t.user_attrs]
        typical = float(np.median(done)) if done else 0.0
        if time.time() + 60 * typical > deadline:
            study.stop()

    study = open_study(cfg, worker=worker)
    max_trials = cfg.get("max_trials")
    callbacks = [stop_before_deadline]
    if max_trials:
        callbacks.append(optuna.study.MaxTrialsCallback(int(max_trials)))
    study.optimize(objective, timeout=max(deadline - time.time(), 1.0), callbacks=callbacks,
                   catch=(RuntimeError, ValueError))


# --- lanzador y resumen -------------------------------------------------------------------

def launch(cfg: dict, config_path: str) -> None:
    import optuna

    study = open_study(cfg)
    if cfg["study"].get("enqueue_base", True):
        enqueue_base(study, cfg)
    deadline = time.time() + 60 * float(cfg["timeout_minutes"])
    folder = out_dir(cfg)
    env = {**os.environ, "OMP_NUM_THREADS": str(cfg["threads_per_worker"]),
           "MKL_NUM_THREADS": str(cfg["threads_per_worker"]), "PYTHONIOENCODING": "utf-8"}
    procs = []
    for w in range(int(cfg["n_workers"])):
        log = open(folder / f"worker{w}.log", "a", encoding="utf-8")  # noqa: SIM115 - vive lo que el proceso
        procs.append((subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--config", config_path,
             "--worker", str(w), "--deadline", str(deadline)],
            cwd=repo_root(), env=env, stdout=log, stderr=subprocess.STDOUT), log))
    logger.info("%d workers × %d hilos | límite %s | logs en %s", len(procs), cfg["threads_per_worker"],
                time.strftime("%H:%M", time.localtime(deadline)), folder)
    while any(p.poll() is None for p, _ in procs):
        time.sleep(60)
        trials = study.get_trials(deepcopy=False)
        states = pd.Series([t.state.name for t in trials]).value_counts().to_dict()
        best = None
        try:
            best = study.best_trial
        except ValueError:
            pass
        logger.info("%s restan | %s%s", f"{max(deadline - time.time(), 0) / 60:4.0f} min", states,
                    f" | mejor t{best.number} = {best.value:.4f}" if best else "")
    for p, log in procs:
        log.close()
        if p.returncode:
            logger.warning("un worker terminó con código %d: ver sus logs", p.returncode)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summarize(cfg)


def summarize(cfg: dict) -> None:
    import optuna

    study = open_study(cfg)
    folder = out_dir(cfg)
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    frame.to_csv(folder / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        logger.warning("ningún trial completo")
        return
    ranked = sorted(complete, key=lambda t: t.value, reverse=True)
    base = next((t for t in study.trials if t.user_attrs.get("is_base")), None)
    budgets = [int(round(100 * float(b))) for b in cfg["objective"]["budgets"]]

    def describe(t) -> dict:
        return {"trial": t.number, "objective": round(float(t.value), 4),
                **{f"det_{b}": round(t.user_attrs[f"mean_det_{b}"], 4) for b in budgets},
                "auc_within_cell": round(t.user_attrs["mean_auc_within_cell"], 4),
                "pr_auc": round(t.user_attrs["mean_pr_auc"], 4),
                "sd_objective_between_repeats": round(t.user_attrs["sd_detection"], 4),
                "minutes": round(t.user_attrs["minutes"], 1), "params": dict(t.params)}

    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    top = {"study": cfg["name"], "n_trials": states, "objective": cfg["objective"],
           "base": describe(base) if base is not None and base.state == optuna.trial.TrialState.COMPLETE else None,
           "top": [describe(t) for t in ranked[: int(cfg["top_k"])]]}
    try:
        top["importances"] = {k: round(float(v), 3) for k, v in
                              optuna.importance.get_param_importances(study).items()}
    except Exception as exc:  # noqa: BLE001 - con pocos trials o espacio condicional puede no salir
        top["importances"] = f"no se pudo calcular: {exc}"
    (folder / "top.yaml").write_text(yaml.safe_dump(top, sort_keys=False, allow_unicode=True), encoding="utf-8")

    cols = ["trial", "objective", *[f"det_{b}" for b in budgets], "auc_within_cell", "pr_auc", "minutes"]
    table = pd.DataFrame([describe(t) for t in ranked[: int(cfg["top_k"])]])[cols]
    print(f"\nTrials: {states}")
    if top["base"]:
        print("Configuración de hoy:", {c: top["base"][c] for c in cols})
    print(table.to_string(index=False))
    print(f"\nImportancias: {top['importances']}")
    print(f"Detalle: {folder / 'top.yaml'} y {folder / 'trials.csv'}")


def write_configs(cfg: dict) -> None:
    """Las top_k (y la de hoy) con semillas del modelo que el sweep no vio, para `train.py`."""
    top = yaml.safe_load((out_dir(cfg) / "top.yaml").read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(resolve_path(cfg["base_config"]).read_text(encoding="utf-8"))
    base = dict(base_cfg["model"]["params"])
    entries = [("base", {})] + [(f"top{i + 1}", e["params"]) for i, e in enumerate(top["top"])]
    written = []
    for tag, params in entries:
        for seed in cfg["confirm_seeds"]:
            name = f"{cfg['name']}-{tag}-s{seed}-r3"
            out = copy.deepcopy(base_cfg)
            out["name"] = name
            out["model"]["params"] = {**to_model_params(base, params), "random_state": int(seed)}
            out["wandb"] = {**out["wandb"], "group": cfg["wandb"]["group"] + "-confirm",
                            "tags": [*cfg["wandb"]["tags"], "confirm"]}
            path = repo_root() / "configs" / f"exp_{name.replace('-', '_')}.yaml"
            header = (f"# Confirmación del sweep de la GRU ({tag}, semilla del modelo {seed}, que el sweep no usó).\n"
                      f"# Generado por scripts/sweep_gru.py --write-configs desde {cfg['name']}/top.yaml.\n"
                      f"#   WANDB_MODE=disabled python scripts/train.py --config {path.relative_to(repo_root()).as_posix()}\n")
            path.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
            written.append(path.relative_to(repo_root()).as_posix())
    print("\n".join(written))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary", action="store_true", help="solo resume el estudio que ya existe")
    parser.add_argument("--write-configs", action="store_true", help="YAMLs de confirmación de las top_k")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M")
    cfg = load_config(args.config)
    if args.worker is not None:
        run_worker(cfg, args.worker, args.deadline)
    elif args.summary:
        summarize(cfg)
    elif args.write_configs:
        write_configs(cfg)
    else:
        launch(cfg, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
