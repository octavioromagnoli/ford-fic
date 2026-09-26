"""El calendario del replay y la política de triage: qué pasa cada semana y qué acción corresponde.

Todo es determinista y sale del bundle y de `policy` en `configs/agents.yaml`. El agente de triage
puede pedir la acción de un auto, pero no elegirla: `check_action` rechaza cualquier otra.

Dos tipos de evento:
- `alerta_nueva`: la alerta sostenida de K2 se confirma (el `k`-ésimo corte seguido sobre el umbral).
  Si el mensaje tiene hábitos que nombrar, la acción es un aviso al conductor; si el riesgo no se
  explica por hábitos, va directo al concesionario (no hay nada que pedirle al conductor).
- `persistencia`: después de un aviso, el score sigue sobre el umbral en las `persist_cuts`
  revisiones siguientes. Se escala al concesionario.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.agents.bundle import Bundle

ALERT, PERSIST = "alerta_nueva", "persistencia"
DRIVER, DEALER = "aviso_conductor", "turno_concesionario"


@dataclass(frozen=True)
class Event:
    vehicle_id: str
    kind: str              # alerta_nueva | persistencia
    date: pd.Timestamp     # cuándo se supo (fecha del corte que lo confirma)
    week: pd.Timestamp     # lunes de esa semana
    action: str            # aviso_conductor | turno_concesionario
    reason: str

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["date"], d["week"] = self.date.isoformat(), self.week.date().isoformat()
        return d


def monday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


def weeks(meta: dict[str, Any]) -> list[pd.Timestamp]:
    """Los lunes del replay, de la semana del inicio a la del fin de la ventana."""
    return list(pd.date_range(monday(meta["replay"]["start"]), monday(meta["replay"]["end"]), freq="7D"))


def week_end(week: pd.Timestamp) -> pd.Timestamp:
    """Fin exclusivo de la semana: el lunes siguiente a las 00:00."""
    return pd.Timestamp(week) + pd.Timedelta(days=7)


def initial_action(vehicle: pd.Series, policy: dict[str, Any]) -> tuple[str, str]:
    if int(vehicle["n_factors"]) > 0:
        return DRIVER, policy["reasons"]["habitos"]
    return DEALER, policy["reasons"]["sin_habitos"]


def persistence_date(cuts: pd.DataFrame, confirm_date: pd.Timestamp, threshold: float, n: int) -> pd.Timestamp | None:
    """Fecha en que se confirma que el riesgo siguió alto `n` revisiones después de la alerta."""
    after = cuts.loc[cuts["cut_date"] > confirm_date].sort_values("cut_odo", kind="stable").head(n)
    if len(after) < n or not after["score"].ge(threshold).all():
        return None
    return pd.Timestamp(after["cut_date"].iloc[-1])


def all_events(bundle: Bundle, policy: dict[str, Any]) -> list[Event]:
    """Todos los eventos del replay, en orden de fecha."""
    events = []
    n = int(policy["persist_cuts"])
    for vid, v in bundle.vehicles.loc[bundle.vehicles["alerted"]].iterrows():
        confirm = pd.Timestamp(v["alert_confirm_date"])
        action, reason = initial_action(v, policy)
        events.append(Event(str(vid), ALERT, confirm, monday(confirm), action, reason))
        if action == DRIVER:
            when = persistence_date(bundle.vehicle_cuts(vid), confirm, bundle.threshold, n)
            if when is not None:
                events.append(Event(str(vid), PERSIST, when, monday(when), DEALER,
                                    policy["reasons"]["persistencia"].format(n=n)))
    return sorted(events, key=lambda e: (e.date, e.vehicle_id))


def events_in_week(events: list[Event], week: pd.Timestamp) -> list[Event]:
    return [e for e in events if e.week == pd.Timestamp(week)]


def expected_action(events: list[Event], vehicle_id: str, week: pd.Timestamp) -> Event | None:
    found = [e for e in events_in_week(events, week) if e.vehicle_id == vehicle_id]
    return found[-1] if found else None


def check_action(events: list[Event], vehicle_id: str, week: pd.Timestamp, proposed: str) -> Event:
    """La acción que el agente propone tiene que ser la de la política. Si no, falla."""
    event = expected_action(events, vehicle_id, week)
    if event is None:
        raise ValueError(f"{vehicle_id} no tiene eventos la semana del {pd.Timestamp(week).date()}")
    if proposed != event.action:
        raise ValueError(f"La política indica `{event.action}` para {vehicle_id}, no `{proposed}`")
    return event
