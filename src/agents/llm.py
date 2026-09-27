"""Cliente de OpenAI con caché en disco: la demo no puede depender del wifi del pitch.

Cada pedido se identifica por el hash de todo lo que lo define (modelo, parámetros, instrucciones,
entrada, esquema o herramientas). Con la misma entrada, la caché devuelve la misma salida, así que un
loop de herramientas completo se reproduce paso a paso sin red. Modos (`llm.mode` o `DEMO_LLM_MODE`):

- `cache_first`: sirve de la caché y llama a la API solo si falta (la app desplegada);
- `cache_only`: nunca llama; si falta, `LLMUnavailable` (y la app muestra la plantilla);
- `live`: siempre llama y no toca la caché (el «regenerar en vivo» de la demo no pisa la versión
  guardada). Para rehacer la caché, borrarla y correr en `cache_first`.

La key sale de `OPENAI_API_KEY` o, en local, del `.env` de la raíz del repo (no se versiona).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import repo_root

MODES = ("cache_first", "cache_only", "live")


class LLMUnavailable(RuntimeError):
    """No hay respuesta en la caché y el modo o la falta de key no permiten llamar a la API."""


def api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env = repo_root() / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


class LLM:
    def __init__(self, cfg: dict[str, Any], cache_dir: Path, *, mode: str | None = None):
        self.cfg = cfg
        self.model = cfg["model"]
        self.mode = mode or os.environ.get("DEMO_LLM_MODE") or cfg.get("mode", "cache_first")
        if self.mode not in MODES:
            raise ValueError(f"modo de LLM `{self.mode}` desconocido: {MODES}")
        self.cache_dir = cache_dir
        self._client = None
        self.calls = {"cache": 0, "api": 0}
        self.used: set[str] = set()   # archivos de la caché que sirvieron o se escribieron en esta sesión

    # -- caché --------------------------------------------------------------------------------------
    def _params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_output_tokens": int(self.cfg["max_output_tokens"])}
        if self.cfg.get("reasoning_effort"):
            params["reasoning"] = {"effort": self.cfg["reasoning_effort"]}
        if self.cfg.get("temperature") is not None:
            params["temperature"] = float(self.cfg["temperature"])
        return params

    def _key(self, request: dict[str, Any]) -> str:
        blob = json.dumps({"model": self.model, **self._params(), **request}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> dict[str, Any] | None:
        if self.mode == "live":
            return None
        path = self.cache_dir / f"{key[:40]}.json"
        if not path.exists():
            return None
        self.used.add(path.name)
        return json.loads(path.read_text(encoding="utf-8"))

    def _put(self, key: str, record: dict[str, Any]) -> None:
        if self.mode == "live":
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key[:40]}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
            self.used.add(f"{key[:40]}.json")
        except OSError:
            pass  # un disco de solo lectura no rompe la demo: la respuesta igual se usa

    @property
    def available(self) -> bool:
        return self.mode != "cache_only" and api_key() is not None

    def _client_or_raise(self):
        if self.mode == "cache_only":
            raise LLMUnavailable("modo cache_only: no se llama a la API")
        if self._client is None:
            key = api_key()
            if not key:
                raise LLMUnavailable("no hay OPENAI_API_KEY (ni en el entorno ni en .env)")
            from openai import OpenAI

            self._client = OpenAI(api_key=key, timeout=float(self.cfg.get("timeout_s", 60)))
        return self._client

    # -- pedidos --------------------------------------------------------------------------------------
    def parse(self, *, instructions: str, payload: str, schema: type[BaseModel]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Salida estructurada (JSON con el esquema de `schema`). Devuelve (salida, metadatos)."""
        request = {"kind": "parse", "instructions": instructions, "input": payload,
                   "schema": schema.model_json_schema()}
        key = self._key(request)
        cached = self._get(key)
        if cached is not None:
            self.calls["cache"] += 1
            return cached["output"], {"cached": True, "model": cached.get("model"), "usage": cached.get("usage")}
        client = self._client_or_raise()
        resp = client.responses.parse(model=self.model, instructions=instructions, input=payload, text_format=schema,
                                      **self._params())
        if resp.output_parsed is None:
            raise RuntimeError(f"El modelo no devolvió una salida con el esquema (estado: {resp.status})")
        out = resp.output_parsed.model_dump()
        usage = resp.usage.model_dump() if resp.usage else None
        self.calls["api"] += 1
        self._put(key, {"request": request, "output": out, "model": resp.model, "usage": usage})
        return out, {"cached": False, "model": resp.model, "usage": usage}

    def respond(self, *, instructions: str, items: list[dict[str, Any]],
                tools: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Un paso de un loop de herramientas. Devuelve {"text", "tool_calls": [{name, arguments, call_id}]}."""
        # `parallel_tool_calls: false` obliga a una herramienta por paso: el agente ve la acción de la política
        # antes de redactar, en vez de adivinarla en el mismo paso.
        extra = {} if self.cfg.get("parallel_tool_calls", True) else {"parallel_tool_calls": False}
        request = {"kind": "respond", "instructions": instructions, "input": items, "tools": tools, **extra}
        key = self._key(request)
        cached = self._get(key)
        if cached is not None:
            self.calls["cache"] += 1
            return cached["output"], {"cached": True, "model": cached.get("model"), "usage": cached.get("usage")}
        client = self._client_or_raise()
        resp = client.responses.create(model=self.model, instructions=instructions, input=items, tools=tools,
                                       **extra, **self._params())
        calls = [{"name": o.name, "arguments": o.arguments, "call_id": o.call_id}
                 for o in resp.output if getattr(o, "type", None) == "function_call"]
        out = {"text": resp.output_text or "", "tool_calls": calls}
        usage = resp.usage.model_dump() if resp.usage else None
        self.calls["api"] += 1
        self._put(key, {"request": request, "output": out, "model": resp.model, "usage": usage})
        return out, {"cached": False, "model": resp.model, "usage": usage}
