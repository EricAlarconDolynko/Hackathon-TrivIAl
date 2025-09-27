# agent/providers.py
import os
from types import SimpleNamespace

# Bedrock (default)
from artifacts.shared.bedrock_client import bedrock_chat

# (Opcional) Cohere fallback si quieres mantenerlo
try:
    import cohere
except Exception:
    cohere = None

def chat_llm(**kwargs):
    """
    Fábrica de chat LLM:
    - Por defecto usa Bedrock (modelo configurable por env).
    - Si LLM_PROVIDER=cohere y hay COHERE_API_KEY, usa Cohere.
    Devuelve un objeto con .invoke(messages) -> SimpleNamespace(content=str)
    """
    provider = (os.getenv("LLM_PROVIDER", "bedrock") or "bedrock").lower()

    if provider == "cohere" and cohere and os.getenv("COHERE_API_KEY"):
        # Adapter mínimo para igualar interfaz .invoke(...)
        api_key = os.environ["COHERE_API_KEY"]
        model = os.getenv("COHERE_CHAT_MODEL", "command-r-plus")
        client = cohere.ClientV2(api_key)

        class CohereChat:
            def __init__(self, model):
                self.model = model
            def invoke(self, messages):
                # messages: lista LC (system/human/ai) o dicts {role, content}
                # Unimos a un único prompt simple (suficiente como fallback)
                joined = []
                for m in messages:
                    role = getattr(m, "type", None) or m.get("role", "user")
                    content = getattr(m, "content", None) or m.get("content", "")
                    joined.append(f"[{role}] {content}")
                text = "\n".join(joined)
                resp = client.chat(model=self.model, messages=[{"role":"user","content":text}])
                out = resp.output_text if hasattr(resp, "output_text") else str(resp)
                return SimpleNamespace(content=out)

        return CohereChat(model)

    # Default: Bedrock
    return bedrock_chat(**kwargs)
