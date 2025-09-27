# agent/providers.py
import os
from typing import Optional
from langchain_cohere import ChatCohere


def chat_cohere(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatCohere:
    """
    Devuelve un cliente ChatCohere listo para usar con LangChain.

    Env vars soportadas:
      - COHERE_API_KEY       (requerida)
      - COHERE_MODEL         (por defecto: "command-r")
      - COHERE_TEMPERATURE   (opcional, float)
      - COHERE_MAX_TOKENS    (opcional, int)
    Parámetros de función tienen prioridad sobre env vars.
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Falta COHERE_API_KEY en variables de entorno.")

    model = model or os.getenv("COHERE_MODEL", "command-r")

    # Permite override por parámetro o por env var
    if temperature is None and (t := os.getenv("COHERE_TEMPERATURE")):
        try:
            temperature = float(t)
        except ValueError:
            pass

    if max_tokens is None and (m := os.getenv("COHERE_MAX_TOKENS")):
        try:
            max_tokens = int(m)
        except ValueError:
            pass

    return ChatCohere(
        model=model,
        cohere_api_key=api_key,
        **({} if temperature is None else {"temperature": temperature}),
        **({} if max_tokens is None else {"max_tokens": max_tokens}),
    )
