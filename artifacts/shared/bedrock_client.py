# lambdas/shared/bedrock_client.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
import boto3
import botocore
from .config import get_settings

# Cache de cliente por cold start
_client = None

def _client_bedrock():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=get_settings().region)
    return _client

def _to_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s)

def _lc_to_bedrock_messages(messages: Union[List[Any], Any]) -> (List[Dict[str, Any]], Optional[List[Dict[str, Any]]]):
    """
    Convierte una lista de mensajes (LangChain o dicts {role,content}) a formato Converse:
      messages = [{"role":"user"|"assistant", "content":[{"text":"..."}]}]
      system = [{"text":"..."}]  (opcional)
    """
    if not isinstance(messages, list):
        messages = [messages]

    system_parts: List[Dict[str, str]] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = None
        content = None

        # LangChain message
        if hasattr(m, "type") and hasattr(m, "content"):
            t = (m.type or "").lower()
            content = _to_text(m.content)
            if t == "system":
                system_parts.append({"text": content})
                continue
            elif t in ("human", "user"):
                role = "user"
            elif t in ("ai", "assistant"):
                role = "assistant"
            else:
                # por seguridad, tratamos como 'user'
                role = "user"

        # dict estilo {"role": "...", "content": "..."}
        elif isinstance(m, dict):
            role = (m.get("role") or "user").lower()
            c = m.get("content")
            if isinstance(c, list):
                # ya puede venir en el formato [{"text": "..."}]
                out.append({"role": role, "content": c})
                continue
            content = _to_text(c)

        else:
            # string plano => usuario
            role = "user"
            content = _to_text(m)

        out.append({"role": role, "content": [{"text": content}]})

    system = system_parts if system_parts else None
    return out, system

def converse_text(messages: Union[List[Any], Any],
                  model_id: Optional[str] = None,
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None,
                  top_p: Optional[float] = None) -> str:
    """
    Llama a Bedrock 'converse' y devuelve el texto del assistant.
    """
    s = get_settings()
    model_id = model_id or s.bedrock_model_id
    temperature = s.llm_temperature if temperature is None else temperature
    max_tokens = s.llm_max_tokens if max_tokens is None else max_tokens
    top_p = s.llm_top_p if top_p is None else top_p

    msgs, system = _lc_to_bedrock_messages(messages)

    req: Dict[str, Any] = {
        "modelId": model_id,
        "messages": msgs,
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
            "topP": float(top_p),
        },
    }
    if system:
        req["system"] = system

    client = _client_bedrock()
    try:
        resp = client.converse(**req)
        # Estructura de salida Converse:
        # resp["output"]["message"]["content"] -> [{"text": "..."}]
        parts = resp.get("output", {}).get("message", {}).get("content", []) or []
        texts = []
        for p in parts:
            if isinstance(p, dict) and "text" in p:
                texts.append(p["text"])
        return "".join(texts).strip()
    except botocore.exceptions.ClientError as e:
        # Exponer info mínima, pero no romper
        raise RuntimeError(f"Bedrock converse error: {e.response.get('Error', {}).get('Message', str(e))}") from e
    except Exception as e:
        raise RuntimeError(f"Bedrock converse unexpected error: {str(e)}") from e

class BedrockChat:
    """
    Cliente sencillo con interfaz .invoke(...) -> SimpleNamespace(content=texto),
    para hacerlo drop-in con el uso que ya tienes en los nodos.
    """
    def __init__(self, model_id: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, top_p: Optional[float] = None):
        self.model_id = model_id or get_settings().bedrock_model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def invoke(self, messages: Union[List[Any], Any]) -> SimpleNamespace:
        text = converse_text(
            messages,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return SimpleNamespace(content=text)

def bedrock_chat(**kwargs) -> BedrockChat:
    """
    Fábrica por si quieres instanciar con overrides:
      bedrock_chat(model_id="...", temperature=0.1, max_tokens=400)
    """
    return BedrockChat(**kwargs)
