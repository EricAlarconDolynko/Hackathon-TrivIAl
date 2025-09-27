# agent/graph.py
from typing import TypedDict, Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from .providers import chat_cohere
from typing import Dict, Any, List
import json, re

# =========================
# Estado del Agente
# =========================
class Symptom(TypedDict, total=False):
    name: str
    severity: Optional[str]          # "leve" | "moderada" | "severa" | None
    frequency: Optional[str]
    onset: Optional[str]
    notes: Optional[str]
    

class AgentState(TypedDict, total=False):
    # Entrada del turno
    user_input: str

    # Demografía (pueden empezar como None)
    age: Optional[int]
    nationality: Optional[str]

    # Captura clínica
    reason_for_visit: Optional[str]
    symptoms: List[Symptom]
    personal_history: Optional[str]
    family_history: Optional[str]

    # Control del flujo conversacional
    missing_fields: List[str]
    needs_another_symptom: bool

    # Conversación
    conversation_context: str
    conversation_response: str

    # Espacio para depuración / telemetría ligera
    meta: Dict[str, Any]
    
# =========================
# Prompt Templates
# =========================

# -------- Prompt Template --------
CONVERSATIONAL_SYSTEM = """Eres un asistente de salud para preconsulta y agendamiento.
Objetivo: recopilar datos iniciales con empatía y claridad; NO diagnostiques.
Reglas:
- PRIMER turno (sin contexto): pregunta primero el MOTIVO DE LA CONSULTA y luego empieza a indagar por SÍNTOMAS.
- Para cada síntoma, captura: nombre, severidad (leve|moderada|severa), frecuencia/periodicidad, inicio/duración y notas relevantes.
- Haz 1 a 2 preguntas por turno máximo. Prioriza completar campos faltantes del síntoma activo antes de pasar a otro.
- Si el usuario menciona más de un síntoma, regístralos por separado y pregunta si hay otro síntoma adicional.
- Cierra con una pregunta clara que ayude a avanzar.
- Adapta el tono según edad/nacionalidad:
  * <18 años: lenguaje muy sencillo y paciente.
  * ≥18: claro y respetuoso.
  * Sé culturalmente sensible.

Al final, ANEXA un bloque <DATA> con JSON estricto:
<DATA>
{{
  "reason_for_visit": "string|null",
  "symptoms": [
    {{
      "name": "string",
      "severity": "leve|moderada|severa|null",
      "frequency": "string|null",
      "onset": "string|null",
      "notes": "string|null"
    }}
  ],
  "personal_history": "string|null",
  "family_history": "string|null",
  "missing_fields": ["string"],
  "needs_another_symptom": true
}}
</DATA>
No incluyas texto fuera del JSON dentro de <DATA>.
"""
# Template con variables contextuales
CONV_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", CONVERSATIONAL_SYSTEM),
    ("system", "Edad conocida: {age}\nNacionalidad conocida: {nationality}\n"),
    ("system", "Contexto previo (resumen corto): {conversation_context}\n"),
    ("human", "{user_turn}")
])


# =========================
# Estado de extracción
# =========================

def _extract_structured_data(text: str) -> Dict[str, Any]:
    data = {
        "reason_for_visit": None,
        "symptoms": [],
        "personal_history": None,
        "family_history": None,
        "missing_fields": [],
        "needs_another_symptom": False,
    }
    m = re.search(r"<DATA>\s*(\{.*?\})\s*</DATA>", text, flags=re.DOTALL)
    if not m:
        return data
    blob = m.group(1)
    for candidate in (blob, blob.strip().strip("`")):
        try:
            parsed = json.loads(candidate)
            for k in data.keys():
                if k in parsed:
                    data[k] = parsed[k]
            break
        except Exception:
            continue
    return data


# =========================
# Nodos (LLMs Cohere)
# =========================

def node_conversational(state: dict) -> dict:
    """
    Conversacional (preconsulta):
    - Primer turno: motivo de consulta y primer síntoma.
    - Turnos siguientes: completa campos faltantes del síntoma activo y/o pregunta por síntomas adicionales.
    - Acumula síntomas en state["symptoms"] (lista de dicts).
    - Señaliza state["needs_another_symptom"] para seguir preguntando en el próximo turno.
    """
    llm = chat_cohere()

    age = state.get("age") or "desconocida"
    nationality = state.get("nationality") or "desconocida"
    context = (state.get("conversation_context") or "").strip()

    # Hint fuerte para el primer turno si no hay contexto ni motivo guardado
    first_turn_hint = ""
    if not context and not state.get("reason_for_visit"):
        first_turn_hint = (
            "Parece ser la primera interacción. Pregunta primero el motivo de la consulta. "
            "Luego identifica el síntoma principal con severidad, frecuencia y comienzo. "
            "Si procede, pregunta por antecedentes personales y familiares de forma breve.\n"
        )

    # Si hay campos faltantes guardados de turnos previos, recuérdalo al LLM
    missing_fields: List[str] = state.get("missing_fields") or []

    # Si el flujo anterior detectó que hay más síntomas, invítalo a nombrar el siguiente
    needs_another_symptom_prev = bool(state.get("needs_another_symptom"))

    guidance = []
    if missing_fields:
        guidance.append(
            "Aún faltan estos datos (prioriza preguntarlos): " + "; ".join(missing_fields)
        )
    if needs_another_symptom_prev:
        guidance.append(
            "El usuario indicó/parece tener más síntomas. Pregunta por el siguiente síntoma y sus detalles."
        )

    user_turn = (first_turn_hint + "\n".join(guidance) + "\n" + (state.get("user_input") or "")).strip()

    prompt = CONV_TEMPLATE.format_messages(
        age=age,
        nationality=nationality,
        conversation_context=context if context else "(sin contexto previo)",
        user_turn=user_turn
    )

    msg = llm.invoke(prompt)
    assistant_text = msg.content

    extracted = _extract_structured_data(assistant_text)

    # Actualizar estado
    new_state = dict(state)

    # Motivo de consulta
    if extracted.get("reason_for_visit"):
        new_state["reason_for_visit"] = extracted["reason_for_visit"]

    # Síntomas (acumular)
    if isinstance(extracted.get("symptoms"), list) and extracted["symptoms"]:
        prev = new_state.get("symptoms") or []
        new_state["symptoms"] = prev + extracted["symptoms"]

    # Antecedentes
    if extracted.get("personal_history") is not None:
        new_state["personal_history"] = extracted["personal_history"]
    if extracted.get("family_history") is not None:
        new_state["family_history"] = extracted["family_history"]

    # Faltantes y bandera para más síntomas
    new_state["missing_fields"] = extracted.get("missing_fields") or []
    new_state["needs_another_symptom"] = bool(extracted.get("needs_another_symptom"))

    # Contexto compacto
    new_context = (context + "\n" + assistant_text).strip() if context else assistant_text.strip()
    if len(new_context) > 1400:
        new_context = new_context[-1400:]
    new_state["conversation_context"] = new_context
    new_state["conversation_response"] = assistant_text

    return new_state

# =========================
# Grafo minimal y runner
# =========================
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("conversational", node_conversational)
    g.set_entry_point("conversational")
    g.add_edge("conversational", END)  # un solo paso para probar
    return g.compile()

_app = build_graph()

def run_agent(user_input: str, age: Optional[int] = None, nationality: Optional[str] = None) -> Dict[str, Any]:
    init: AgentState = {
        "user_input": user_input,
        "age": age,
        "nationality": nationality,
        "reason_for_visit": None,
        "symptoms": [],
        "personal_history": None,
        "family_history": None,
        "missing_fields": [],
        "needs_another_symptom": False,
        "conversation_context": "",
        "conversation_response": "",
    }
    result = _app.invoke(init)
    return {
        "reply": result.get("conversation_response", ""),
        "state": {
            "reason_for_visit": result.get("reason_for_visit"),
            "symptoms": result.get("symptoms"),
            "personal_history": result.get("personal_history"),
            "family_history": result.get("family_history"),
            "missing_fields": result.get("missing_fields"),
            "needs_another_symptom": result.get("needs_another_symptom"),
        }
    }