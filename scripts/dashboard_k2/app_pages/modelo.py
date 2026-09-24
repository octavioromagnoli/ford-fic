"""Modelo: K2 contra la referencia y los pisos, qué mide cada número y qué no hay que prometer."""

import numpy as np
import pandas as pd
import streamlit as st

from scripts.dashboard_k2.common import cfg, data, dec, label
from src.eval.dashboard_data import LABEL_NAMES

d, settings = data(), cfg()
lab = label()

# --- comparación (window_eval.json, la misma cuenta que decide el finalista) -----------------
with st.container(border=True):
    st.markdown(f"**K2 contra el finalista anterior y los pisos** · etiqueta {LABEL_NAMES[lab].lower()}, 5% de falsas alarmas")
    if d.window_eval is None:
        st.info("Falta `window_eval.json`. Corré `python scripts/eval_window_label.py --config "
                f"{settings['run_config']}`.")
    else:
        res = d.window_eval["results"][lab]
        entries = [("K2 (finalista)", res["candidate"]), ("Finalista anterior", res["reference"]),
                   *[(f"Piso: solo `{name}`", m) for name, m in res["floors"].items()]]
        table = pd.DataFrame([{
            "modelo": name,
            "detection": float(np.mean(m["detection"])),
            "sd": float(np.std(m["detection"])),
            "detected": " / ".join(map(str, m["n_detected"])),
            "lift": float(np.mean(m["vehicle_lift_mean"])),
            "lead_km": float(np.nanmean(m["lead_km"])) if len(m["lead_km"]) and np.isfinite(m["lead_km"]).any() else np.nan,
        } for name, m in entries])
        st.dataframe(table, hide_index=True, column_config={
            "modelo": "Modelo",
            "detection": st.column_config.NumberColumn("Detección", format="percent"),
            "sd": st.column_config.NumberColumn("± entre repeticiones", format="percent"),
            "detected": "Detectados por repetición",
            "lift": st.column_config.NumberColumn("Lift por vehículo", format="%.2f×",
                                                  help="Cuánto mejor que el azar ordena a los autos (score medio por auto)."),
            "lead_km": st.column_config.NumberColumn("Anticipación mediana [km]", format="%,.0f"),
        })
        st.caption("Los pisos usan una sola columna como score: el odómetro del corte o los días desde la venta. "
                   "K2 les gana a los dos. La mejora sobre el finalista anterior es de 2–3 autos y el bootstrap por "
                   f"vehículo no la separa del cero (`{settings['docs']['finalist']}`).")

# --- métricas por fila (metrics.json, etiqueta dura) ----------------------------------------
if d.metrics:
    m = d.metrics
    with st.container(border=True):
        st.markdown("**Qué sabe el modelo: *qué auto* más que *cuándo*** · por fila, etiqueta dura")
        with st.container(horizontal=True):
            st.metric("PR-AUC por fila", dec(m["oof"]["pr_auc"]),
                      f"techo de cohorte {dec(m['cohort_ceiling']['pr_auc'])}", delta_color="off", border=True,
                      help="Puntuar cada fila con '¿este auto falla?', sin nada del cuándo, ya da el techo de cohorte. "
                           "Por debajo de él, un PR-AUC por fila no demuestra anticipación (regla 6).")
            st.metric("Aporte del *cuándo* (a′)", dec(m["when_spread"]["when_delta"]["mean"], sign=True),
                      f"± {dec(m['when_spread']['when_delta']['std'])}", delta_color="off", border=True,
                      help="Cuánto cae el PR-AUC al reemplazar el score de cada fila por el promedio de su auto.")
            st.metric("C-index fuera de fold", dec(m["concordance"]["c_index"]), border=True)
            st.metric("Brier", dec(m["oof"]["brier"]), f"tasa base {dec(m['oof']['base_rate'])}",
                      delta_color="off", border=True)
        st.caption("El rasgo que predice es del auto y está desde los primeros meses; no hay una degradación "
                   "visible que se acelere antes del evento. Por eso la alerta llega meses antes, y por eso no "
                   "se puede afinar mucho el momento.")

if d.audit is None:
    st.caption("Las auditorías obligatorias (a0, a′, b) no están en este checkout: "
               f"`python scripts/audit_model.py --config {settings['run_config']}`. Sus números están en "
               f"`{settings['docs']['finalist']}`.")

# --- límites ----------------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Lo que hay que decir junto con el número**")
    st.markdown(
        f"""
- **Son 45 autos que fallan y 95 sanos en dev.** Un auto son ~2 puntos de detección, así que el número del
  pitch es un rango (**~15–17%** a 5% de falsas alarmas), no un punto (`{settings['docs']['variance']}`).
- **Por debajo de 5% de falsas alarmas, K2 no le gana al azar.** No se prometen alertas "casi sin falsas alarmas".
- **Todo se mide en dev.** El test (74 autos) no se tocó: se mide una sola vez, con el modelo ya elegido.
- **Vale para CNTRY_3 y CNTRY_4.** Son los mercados donde el evento tiene fecha. En los otros, el rasgo
  temprano no se replicó.
- **Para detectar más hacen falta más eventos, no otro modelo.** Eso depende de que Ford explique la fecha por
  defecto de `IdentificationDate` (`docs/memoria/f8-datar-eventos-fase0.md`).
"""
    )
