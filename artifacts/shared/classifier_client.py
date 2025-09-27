# lambdas/shared/classifier_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from urllib import request, error
import json
import time
from .config import get_settings
from .logger import log

class ClassifierError(RuntimeError):
    pass

def _http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout_s: int = 8) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            if v is not None:
                req.add_header(k, v)

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise ClassifierError(f"Classifier HTTP {e.code}: {body}")
    except error.URLError as e:
        raise ClassifierError(f"Classifier connection error: {e.reason}")
    except Exception as e:
        raise ClassifierError(f"Classifier unknown error: {str(e)}")

def _normalize_differentials(obj: Any) -> List[Dict[str, Any]]:
    """
    Acepta varias formas de respuesta, devuelve lista de:
      [{"condition": str, "probability": float, "rationale": str}, ...][:3]
    """
    diffs = []
    # casos comunes: {"differentials": [...] } o {"results":[...]} o lista directa
    if isinstance(obj, dict):
        if isinstance(obj.get("differentials"), list):
            diffs = obj["differentials"]
        elif isinstance(obj.get("results"), list):
            diffs = obj["results"]
        elif isinstance(obj.get("data"), list):
            diffs = obj["data"]
        else:
            # buscar el primer array en el dict
            for v in obj.values():
                if isinstance(v, list):
                    diffs = v
                    break
    elif isinstance(obj, list):
        diffs = obj

    out: List[Dict[str, Any]] = []
    for d in (diffs or []):
        try:
            cond = str(d.get("condition", "")).strip()
            prob = float(d.get("probability", 0.0))
            rat  = str(d.get("rationale", "")).strip()
            if not cond:
                continue
            # recorte/validación básica
            if prob < 0.0: prob = 0.0
            if prob > 1.0: prob = 1.0
            out.append({"condition": cond, "probability": prob, "rationale": rat})
        except Exception:
            continue
        if len(out) >= 3:
            break
    return out

def classify_case(case_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Envía el caso al clasificador externo y devuelve top-3 diferenciales normalizados.
    - Lee URL/key/timeout de env vars.
    - Lanza ClassifierError en caso de error.
    """
    s = get_settings()
    if not s.classifier_api_url:
        raise ClassifierError("CLASSIFIER_API_URL is not set")

    headers = {}
    if s.classifier_api_key:
        headers["Authorization"] = f"Bearer {s.classifier_api_key}"

    timeout_s = max(1, int(s.classifier_timeout_ms / 1000))

    t0 = time.time()
    resp = _http_post_json(
        s.classifier_api_url,
        payload={"case": case_json},
        headers=headers,
        timeout_s=timeout_s,
    )
    dt = int((time.time() - t0) * 1000)
    log.info("classifier_call", url=s.classifier_api_url, ms=dt)

    diffs = _normalize_differentials(resp)
    return diffs
