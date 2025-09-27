# agent/graph.py
from typing import TypedDict, Literal, Optional, Dict, Any, Tuple
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate    
from .providers import chat_cohere
from typing import Dict, Any, List, Optional
import json, re

# =========================
# Estado del Agente
# =========================

class Habit(TypedDict, total=False):
    name: str
    regularidad: Optional[str] 
class Symptom(TypedDict, total=False):
    name: str
    severity: Optional[str]          # "leve" | "moderada" | "severa" | None
    frequency: Optional[str]
    onset: Optional[str]
    notes: Optional[str]
class Differential(TypedDict, total=False):
    condition: str
    probability: float
    rationale: str

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
    
    # Llenar espacios 
    pending_slots: List[Dict[str, Any]]       # cola de huecos por llenar
    last_asked_slot: Optional[Dict[str, Any]]
    
    # Hábitos
    habits: List[Habit]
    habits_suggested: List[str]
    pending_habit_slots: List[Dict[str, Any]]
    last_asked_habit_slot: Optional[Dict[str, Any]]
    missing_habits_fields: List[str]

    # Juzgar    
    case_en: Dict[str, Any]                 # JSON del caso en inglés (normalizado)
    differentials: List[Differential]       # top-3 diferenciales

    # Señal del conversacional para ruteo
    conv_done: bool
    conv_route: Literal["ask_user", "to_symptom", "to_habits"]  # puedes usar sólo ask_user / to_habits si prefieres
    symptom_done: bool
    symptom_route: Literal["ask_user", "to_habits"]
    classifier_done: bool
    classifier_route: Literal["to_end", "to_judge"]
    
# =========================
# Prompt Templates
# =========================

# -------- Prompt Template --------
CONVERSATIONAL_SYSTEM = """Eres un asistente de salud para preconsulta y agendamiento.
Objetivo: recopilar datos iniciales con empatía y claridad; NO diagnostiques.
Reglas:
- PRIMER turno (sin contexto): pregunta primero el MOTIVO DE LA CONSULTA y luego empieza a indagar por SÍNTOMAS.
- Para cada síntoma, captura: nombre, severidad (leve|moderada|severa), frecuencia/periodicidad, inicio/duración y notas relevantes.
- Haz EXACTAMENTE 1 PREGUNTA por turno. No enumeres varias preguntas.
- Si el usuario menciona más de un síntoma, regístralos por separado y pregunta si hay otro síntoma adicional (de una en una, turno a turno).
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


SYMPTOM_VALIDATOR_SYSTEM = """Eres un asistente de salud. Tu tarea es redactar UNA sola pregunta,
breve, clara y empática para completar un dato clínico faltante. No diagnostiques ni indiques
tratamientos. Si la persona no sabe, debe poder responder “no sé”."""

SYMPTOM_VALIDATOR_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYMPTOM_VALIDATOR_SYSTEM),
    ("human",
     "Edad: {age}\nNacionalidad: {nationality}\n"
     "Contexto previo (resumen breve): {context}\n\n"
     "Campo a completar: {slot_label}\n"
     "Base de la pregunta (guía): {base_question}\n\n"
     "Redacta una sola pregunta natural y amable usando la guía.\n"
     "Incluye que si no sabe, puede responder “no sé”. No agregues nada más.")
])

HABITS_SYSTEM = """Eres un asistente de salud. Tu tarea es redactar de forma breve,
clara y empática preguntas sobre hábitos relacionados con síntomas previos.
No diagnostiques ni des tratamientos. Haz como máximo 1 pregunta en este turno.
Si el usuario no sabe, debe poder responder “no sé”."""


HABITS_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", HABITS_SYSTEM),
    ("human",
     "Edad: {age}\nNacionalidad: {nationality}\n"
     "Contexto previo (resumen breve): {context}\n\n"
     "Introduce brevemente (si aplica): {intro}\n"
     "Formula en tono adecuado estas preguntas (1 máximo), fusionadas en un único mensaje natural:\n"
     "{questions}\n\n"
     "Cierra con una frase que invite a responder. No agregues nada más.")
])

CLASSIFIER_SYSTEM = """You are a careful medical triage assistant. Translate the provided Spanish case into
clear ENGLISH JSON and produce a calibrated differential diagnosis (top-3). Do NOT give treatment or final diagnosis.
Return ONLY JSON with the exact schema below.

Schema (JSON):
{
  "case_en": {
    "age": number|null,
    "nationality": "string|null",
    "chief_complaint": "string|null",
    "symptoms": [
      {
        "name": "string|null",
        "severity": "mild|moderate|severe|null",
        "frequency": "string|null",
        "onset": "string|null",
        "notes": "string|null"
      }
    ],
    "personal_history": "string|null",
    "family_history": "string|null",
    "habits": [
      {"name": "string", "regularity": "string|null"}
    ]
  },
  "differentials": [
    {"condition": "string", "probability": 0.0, "rationale": "string"},
    {"condition": "string", "probability": 0.0, "rationale": "string"},
    {"condition": "string", "probability": 0.0, "rationale": "string"}
  ]
}

Constraints:
- Translate content to ENGLISH in "case_en".
- Probabilities in [0,1], sum ≈ 1.0 (allow minor rounding).
- Be conservative; this is NOT a diagnosis.
"""

CLASSIFIER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", CLASSIFIER_SYSTEM),
    ("human",
     "SPANISH CASE JSON:\n{case_es}\n\n"
     "Return ONLY the JSON following the schema. No extra text.")
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


def _render_habits_text(age, nationality, context, intro, questions_list):
    """
    Redacta el texto final (bonito) con Cohere.
    Tiene fallback determinista si hay errores o rate limit.
    """
    llm = chat_cohere()
    msgs = HABITS_TEMPLATE.format_messages(
        age = age if age is not None else "desconocida",
        nationality = nationality or "desconocida",
        context = context or "(sin contexto)",
        intro = intro or "",
        questions = "\n".join(f"- {q}" for q in questions_list if q.strip())
    )
    try:
        return llm.invoke(msgs).content
    except Exception:
        prefix = (intro + "\n") if intro else ""
        return prefix + " ".join(questions_list)
    
def _build_patient_case_json_es(state: "AgentState") -> Dict[str, Any]:
    """Arma un resumen del caso (en español) con lo que hay en el estado."""
    return {
        "age": state.get("age"),
        "nationality": state.get("nationality"),
        "chief_complaint": state.get("reason_for_visit"),
        "symptoms": state.get("symptoms") or [],
        "personal_history": state.get("personal_history"),
        "family_history": state.get("family_history"),
        "habits": state.get("habits") or [],
    }

def _json_from_text(text: str) -> Dict[str, Any]:
    """Extrae el primer bloque JSON del texto (tolerante a ruido)."""
    try:
        # intento directo
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            # limpia backticks, comas colgantes comunes
            cleaned = blob.strip().strip("`")
            try:
                return json.loads(cleaned)
            except Exception:
                return {}
    return {}

# =========================
# Nodos (LLMs Cohere)
# =========================

import re

# --- Helpers ---

def _strip_data_block(text: str) -> str:
    """Quita <DATA>{...}</DATA> antes de mostrar al usuario."""
    if not text:
        return text
    cleaned = re.sub(r"<DATA>\s*{.*?}\s*</DATA>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned

def _merge_symptoms(prev_list, incoming_list):
    """
    Une síntomas por nombre (normalizado) y SOLO completa campos vacíos.
    No pisa valores ya presentes; si llega sin nombre, lo agrega como anónimo.
    """
    def norm(s): return (s or "").strip().lower()
    by_name = {norm(s.get("name")): dict(s) for s in (prev_list or []) if s}
    anon = 0
    for inc in (incoming_list or []):
        if not inc:
            continue
        key = norm(inc.get("name"))
        if not key:
            by_name[f"__anon__{anon}"] = {k: v for k, v in inc.items() if v not in (None, "", "null")}
            anon += 1
            continue
        base = by_name.get(key, {})
        for field, val in inc.items():
            if val in (None, "", "null"):
                continue
            if base.get(field) in (None, "", "null"):
                base[field] = val
        if not base.get("name"):
            base["name"] = inc.get("name")
        by_name[key] = base
    return list(by_name.values())

def _user_denies_more_symptoms(text: str) -> bool:
    t = (text or "").strip().lower()
    negatives = [
        "no tengo otros sintomas", "no tengo otros síntomas",
        "ningun otro", "ningún otro", "no hay otros",
        "no mas sintomas", "no más síntomas",
        "solo ese", "solo esa", "solo eso",
        "no presento otros", "no presento otros sintomas", "no presento otros síntomas"
    ]
    return any(pat in t for pat in negatives)

# --- Nodo conversacional ---

def node_conversational(state: dict) -> dict:
    """
    Conversacional:
    - Habla con el usuario para listar síntomas (1 pregunta por turno).
    - No calcula faltantes; sólo registra motivo y síntomas.
    - Conv_done = True cuando el usuario ya LISTÓ todos sus síntomas (p. ej. niega más).
    - Si hay pendientes de symptom/habits, hace passthrough (no habla) y cede.
    - Quita <DATA> de la respuesta visible.
    """
    new_state = dict(state)

    # 0) Passthrough si ya hay pendientes abajo (evita dos voces)
    if new_state.get("last_asked_slot") or (new_state.get("pending_slots") or []):
        new_state["conv_done"] = True
        return new_state  # route_from_conversational -> to_symptom
    if new_state.get("last_asked_habit_slot") or (new_state.get("pending_habit_slots") or []):
        new_state["conv_done"] = True
        return new_state  # route -> to_habits (si lo manejas en otro router)

    llm = chat_cohere()

    age = state.get("age") or "desconocida"
    nationality = state.get("nationality") or "desconocida"
    context = (state.get("conversation_context") or "").strip()

    # ¿Es el turno inicial (el agente abre)?
    initial_agent_turn = not bool(state.get("user_input"))
    first_turn = not (state.get("conversation_context") or state.get("reason_for_visit") or state.get("symptoms"))

    # hint solo en primer turno
    first_turn_hint = ""
    if first_turn:
        first_turn_hint = (
            "Parece ser la primera interacción. Pregunta primero el motivo de la consulta. "
            "Luego identifica el síntoma principal (nombre, severidad, frecuencia, inicio). "
            "Recuerda: SOLO 1 pregunta en este turno.\n"
        )

    # guía mínima (NO uses missing_fields aquí para evitar loops)
    needs_more_prev = bool(state.get("needs_another_symptom"))
    denies_more = _user_denies_more_symptoms(state.get("user_input") or "")
    guidance = []
    if needs_more_prev and not denies_more:
        guidance.append("El usuario podría tener más síntomas. Pide UNO nuevo, con una sola pregunta.")

    user_turn = (first_turn_hint + ("\n".join(guidance) + "\n" if guidance else "") + (state.get("user_input") or "")).strip()

    prompt = CONV_TEMPLATE.format_messages(
        age=age,
        nationality=nationality,
        conversation_context=context if context else "(sin contexto previo)",
        user_turn=user_turn
    )

    msg = llm.invoke(prompt)
    raw_text = msg.content
    extracted = _extract_structured_data(raw_text)  # reutiliza tu extractor

    # 1) Registrar motivo y síntomas (merge seguro)
    if extracted.get("reason_for_visit"):
        new_state["reason_for_visit"] = extracted["reason_for_visit"]

    if isinstance(extracted.get("symptoms"), list) and extracted["symptoms"]:
        prev = new_state.get("symptoms") or []
        new_state["symptoms"] = _merge_symptoms(prev, extracted["symptoms"])

    if extracted.get("personal_history") is not None:
        new_state["personal_history"] = extracted["personal_history"]
    if extracted.get("family_history") is not None:
        new_state["family_history"] = extracted["family_history"]

    # 2) Determinar si el usuario YA terminó de listar síntomas
    #    Criterio:
    #    - Si NIEGA explícitamente -> terminado.
    #    - Si no niega, usa la señal del modelo: needs_another_symptom==True => no terminado.
    #      Si la señal viene ausente/None, asumimos que AÚN NO terminó (para no cortar antes).
    model_flag = extracted.get("needs_another_symptom")
    if initial_agent_turn:
        # el primer turno del agente nunca cede inmediatamente
        conv_done = False
        needs_more = True
    else:
        if denies_more:
            conv_done = True
            needs_more = False
        else:
            needs_more = True if model_flag is None else bool(model_flag)
            conv_done = not needs_more

    new_state["needs_another_symptom"] = needs_more
    new_state["conv_done"] = conv_done
    # Nota: tu router ya usa sólo conv_done

    # 3) Mostrar texto (sin <DATA>) sólo si NO vamos a ceder ya
    visible_text = _strip_data_block(raw_text)
    if not conv_done:
        prev_ctx = state.get("conversation_context") or ""
        new_ctx = (prev_ctx + ("\n" if prev_ctx else "") + visible_text).strip()
        if len(new_ctx) > 1400:
            new_ctx = new_ctx[-1400:]
        new_state["conversation_context"] = new_ctx
        new_state["conversation_response"] = visible_text
    # si conv_done=True, no escribimos respuesta; hablará symptom en este mismo invoke

    return new_state


# =========================
# Validador de Sintomas 
# =========================


_REQUIRED_SYMPTOM_FIELDS = ("name", "severity", "frequency", "onset", "notes")
_NO_INFO_TEXT = "No hay conocimiento sobre este aspecto"


def _render_symptom_question(age, nationality, context, slot_label, base_question) -> str:
    llm = chat_cohere()
    msgs = SYMPTOM_VALIDATOR_TEMPLATE.format_messages(
        age = age if age is not None else "desconocida",
        nationality = nationality or "desconocida",
        context = (context or "").strip() or "(sin contexto)",
        slot_label = slot_label,
        base_question = base_question,
    )
    try:
        return llm.invoke(msgs).content
    except Exception:
        # Fallback determinista si hay error/rate limit
        return f"{base_question} Si no sabes o no aplica, puedes decir “no sé”."

# --- Helpers de parsing/slots (deterministas) ---
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _is_no_info_answer(user_text: str) -> bool:
    """Detecta respuestas tipo 'no sé', 'no se', 'no recuerdo', 'no tengo', 'n/a', etc."""
    t = _normalize_text(user_text)
    triggers = ("no se", "no sé", "no recuerdo", "no tengo", "n/a", "ninguno", "ninguna", "no aplica", "no aplica.", "no")
    return any(tok in t for tok in triggers)

def _slot_key(slot: Dict[str, Any]) -> str:
    """Clave única para un slot (para historial)."""
    if slot.get("scope") == "top":
        return f"top::{slot['field']}"
    return f"sym::{slot.get('index','?')}::{slot['field']}"

def _compute_pending_slots(state: "AgentState") -> List[Dict[str, Any]]:
    """Devuelve lista de slots faltantes: primero top-level, luego por síntoma."""
    slots: List[Dict[str, Any]] = []

    # Top-level
    if state.get("reason_for_visit") in (None, "", "null"):
        slots.append({"scope": "top", "field": "reason_for_visit"})

    if state.get("personal_history") in (None, "", "null"):
        slots.append({"scope": "top", "field": "personal_history"})

    if state.get("family_history") in (None, "", "null"):
        slots.append({"scope": "top", "field": "family_history"})

    # Síntomas
    symptoms = state.get("symptoms") or []
    for i, sym in enumerate(symptoms):
        sym = sym or {}
        name = sym.get("name")
        for f in _REQUIRED_SYMPTOM_FIELDS:
            val = sym.get(f)
            if val in (None, "", "null"):
                slots.append({
                    "scope": "symptom",
                    "index": i,
                    "symptom_name": name or "(síntoma sin nombre)",
                    "field": f
                })
    return slots

def _set_slot_value_in_state(state: "AgentState", slot: Dict[str, Any], value: str) -> None:
    """Escribe el valor en el state según el slot (top-level o síntoma)."""
    if slot.get("scope") == "top":
        state[slot["field"]] = value
        return
    if slot.get("scope") == "symptom":
        idx = slot.get("index", 0)
        symptoms = state.get("symptoms") or []
        if idx >= len(symptoms):
            return
        if symptoms[idx] is None:
            symptoms[idx] = {}
        symptoms[idx][slot["field"]] = value
        state["symptoms"] = symptoms

def _pretty_slot(slot: Dict[str, Any]) -> str:
    """Texto humano corto del slot (para missing_fields)."""
    if slot.get("scope") == "top":
        return slot["field"]
    return f"{slot['field']}: {slot.get('symptom_name','(desconocido)')}"

def _question_for_slot(slot: Dict[str, Any]) -> str:
    """Pregunta base, determinista, que luego el template “viste”."""
    if slot.get("scope") == "top":
        f = slot["field"]
        if f == "reason_for_visit":
            return "¿Cuál es el motivo principal de tu consulta hoy?"
        if f == "personal_history":
            return "¿Tienes antecedentes personales relevantes (alergias, cirugías, enfermedades crónicas)?"
        if f == "family_history":
            return "¿Hay antecedentes familiares importantes (por ejemplo: diabetes, cardiopatías, cáncer)?"
        return f"¿Puedes especificar {f}?"
    else:
        fname = slot["field"]
        sname = slot.get("symptom_name") or "el síntoma"
        if fname == "name":
            return "Mencionaste otro malestar. ¿Cómo se llama ese síntoma?"
        if fname == "severity":
            return f"¿Qué tan intenso es {sname}? (leve, moderada o severa)"
        if fname == "frequency":
            return f"¿Con qué frecuencia aparece {sname}? (por ejemplo: diario, 2-3 veces por semana)"
        if fname == "onset":
            return f"¿Cuándo comenzó {sname}? (por ejemplo: hace 3 días, desde anoche)"
        if fname == "notes":
            return f"¿Hay algo adicional sobre {sname} que quieras agregar?"
        return f"¿Podrías detallar {fname} para {sname}?"

def _maybe_interpret_direct_answer(slot: Dict[str, Any], user_text: str) -> Optional[str]:
    """
    Intenta mapear la respuesta del usuario directamente al slot.
    - Si 'no sé' => _NO_INFO_TEXT
    - severity: detecta leve|moderada|severa
    - en otros campos: devuelve el texto crudo si hay algo; si no, None
    """
    if _is_no_info_answer(user_text):
        return _NO_INFO_TEXT

    t = _normalize_text(user_text)
    if slot.get("scope") == "symptom" and slot.get("field") == "severity":
        if "leve" in t: return "leve"
        if "moderad" in t: return "moderada"
        if "sever" in t or "intens" in t or "fuerte" in t: return "severa"

    raw = user_text.strip()
    return raw if raw else None

def _auto_fill_prev_asked_unknown(state: "AgentState", pending: List[Dict[str, Any]]) -> None:
    """
    Si un slot ya fue preguntado antes (historial) y aún sigue pendiente,
    lo marcamos automáticamente como 'No hay conocimiento...' para NO insistir.
    """
    asked_hist = state.get("asked_slots_history") or []
    asked_keys = { rec.get("key") for rec in asked_hist if rec.get("key") }

    changed = False
    for slot in list(pending):
        key = _slot_key(slot)
        if key in asked_keys:
            _set_slot_value_in_state(state, slot, _NO_INFO_TEXT)
            changed = True

    if changed:
        # Si rellenamos automáticamente, conviene recalcular pending
        new_pending = _compute_pending_slots(state)
        pending[:] = new_pending  # muta la lista recibida


def node_symptom_validator(state: "AgentState") -> "AgentState":
    new_state: AgentState = dict(state)
    user_text = state.get("user_input") or ""

    if new_state.get("asked_slots_history") is None:
        new_state["asked_slots_history"] = []

    # Completar el slot previo si había
    last_slot = state.get("last_asked_slot")
    if last_slot:
        val = _maybe_interpret_direct_answer(last_slot, user_text)
        if val is None:
            val = _NO_INFO_TEXT
        _set_slot_value_in_state(new_state, last_slot, val)
        new_state["asked_slots_history"].append({
            "key": _slot_key(last_slot),
            "slot": last_slot,
            "status": "answered_unknown" if val == _NO_INFO_TEXT else "answered_value",
            "value": val,
        })
        new_state["last_asked_slot"] = None

    # Recalcular pendientes
    pending = _compute_pending_slots(new_state)
    _auto_fill_prev_asked_unknown(new_state, pending)   # no insistir
    new_state["pending_slots"] = pending
    new_state["missing_fields"] = [_pretty_slot(s) for s in pending]

    # Decisión
    if not pending:
        new_state["symptom_done"] = True
        new_state["symptom_route"] = "to_habits"
        # ⬇⬇ sobrescribe, NO concatena
        new_state["conversation_response"] = "Gracias. Ya tengo completos los datos principales de tus síntomas."
        return new_state

    new_state["symptom_done"] = False
    new_state["symptom_route"] = "ask_user"

    # 1 sola pregunta por turno
    slot = pending[0]
    base_q = _question_for_slot(slot)
    slot_label = _pretty_slot(slot)

    age = new_state.get("age")
    nationality = new_state.get("nationality")
    context = new_state.get("conversation_context")
    pretty_q = _render_symptom_question(age, nationality, context, slot_label, base_q)

    new_state["last_asked_slot"] = slot
    new_state["asked_slots_history"].append({
        "key": _slot_key(slot),
        "slot": slot,
        "status": "asked",
        "value": None,
    })

    # ⬇⬇ sobrescribe, NO concatena
    new_state["conversation_response"] = pretty_q

    return new_state


# =========================
# Habitos
# =========================


_NO_INFO_TEXT = "No hay conocimiento sobre este aspecto"

# Mapa simple: keywords de síntomas -> hábitos sugeridos
_HABIT_SUGGESTIONS = {
    "cabeza":   ["sueño", "hidratación", "cafeína", "pantallas", "estrés", "alcohol", "tabaco"],
    "cefalea":  ["sueño", "hidratación", "cafeína", "pantallas", "estrés"],
    "migra":    ["sueño", "hidratación", "cafeína", "estrés"],
    "fiebre":   ["hidratación", "higiene_manos"],
    "tos":      ["tabaco", "vapeo", "hidratación", "exposición_ambiental"],
    "garganta": ["hidratación", "tabaco"],
    "dolor":    ["actividad_física", "sueño", "hidratación"],
    "abdomen":  ["alimentación", "alcohol", "cafeína", "medicación", "hidratación"],
    "estómago": ["alimentación", "cafeína", "alcohol", "hidratación"],
    "diarrea":  ["hidratación", "alimentación"],
    "estreñ":   ["hidratación", "fibra/alimentación", "actividad_física"],
    "ansiedad": ["estrés", "sueño", "cafeína", "actividad_física"],
    "insomnio": ["sueño", "pantallas", "cafeína"],
}
_DEFAULT_HABITS = ["sueño", "hidratación", "alimentación", "actividad_física", "cafeína", "alcohol", "tabaco", "medicación", "estrés"]

def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _is_no_info_answer(user_text: str) -> bool:
    t = _normalize_text(user_text)
    return any(tok in t for tok in ("no se", "no sé", "no recuerdo", "no tengo", "n/a", "ninguno", "ninguna", "no aplica"))

def _suggest_habits_from_symptoms(symptoms: List[Dict[str, Any]]) -> List[str]:
    names = []
    for s in symptoms or []:
        n = _normalize_text((s or {}).get("name"))
        if n:
            names.append(n)

    suggested = set()
    for n in names:
        matched = False
        for kw, habits in _HABIT_SUGGESTIONS.items():
            if kw in n:
                suggested.update(habits)
                matched = True
        if not matched:
            suggested.update(_DEFAULT_HABITS)

    if not suggested:
        suggested.update(_DEFAULT_HABITS)

    ordered = [h for h in _DEFAULT_HABITS if h in suggested]
    for habits in _HABIT_SUGGESTIONS.values():
        for h in habits:
            if h in suggested and h not in ordered:
                ordered.append(h)
    return ordered

def _ensure_habits_initialized(state: "AgentState", habits_list: List[str]) -> None:
    current = state.get("habits") or []
    names_present = {_normalize_text(h.get("name")) for h in current}
    for name in habits_list:
        if _normalize_text(name) not in names_present:
            current.append({"name": name, "regularidad": None})
    state["habits"] = current

def _compute_pending_habit_slots(state: "AgentState") -> List[Dict[str, Any]]:
    pending = []
    for i, h in enumerate(state.get("habits") or []):
        if (h or {}).get("regularidad") in (None, "", "null"):
            pending.append({"index": i, "field": "regularidad", "habit_name": h.get("name", "(hábito)")})
    return pending

def _question_for_habit_slot(slot: Dict[str, Any]) -> str:
    hname = slot.get("habit_name", "este hábito")
    return (
        f"Sobre {hname}, ¿con qué regularidad lo mantienes? "
        "Por ejemplo: “varias veces al día”, “diario”, “3-4 veces por semana”, “ocasional”, “nunca”. "
        "Si no sabes o no aplica, puedes decir “no sé”."
    )

def _set_habit_slot_value(state: "AgentState", slot: Dict[str, Any], value: str) -> None:
    idx = slot.get("index", 0)
    habits = state.get("habits") or []
    if 0 <= idx < len(habits):
        if habits[idx] is None:
            habits[idx] = {}
        habits[idx]["regularidad"] = value
        state["habits"] = habits


def node_habits(state: "AgentState") -> "AgentState":
    """
    - Propone hábitos relevantes según los síntomas.
    - Asegura entradas en state['habits'] con regularidad=None.
    - Si hay 'last_asked_habit_slot', intenta llenar con la respuesta del usuario
      (o con 'No hay conocimiento...' si dice 'no sé').
    - Formula 1a2 preguntas pendientes y redacta con el prompt template.
    """
    new_state: AgentState = dict(state)
    user_text = state.get("user_input") or ""

    # 1) Completar el último slot preguntado si aplica
    last_h_slot = state.get("last_asked_habit_slot")
    if last_h_slot:
        if _is_no_info_answer(user_text):
            _set_habit_slot_value(new_state, last_h_slot, _NO_INFO_TEXT)
            new_state["last_asked_habit_slot"] = None
        elif user_text.strip():
            _set_habit_slot_value(new_state, last_h_slot, user_text.strip())
            new_state["last_asked_habit_slot"] = None

    # 2) Sugerir hábitos a partir de síntomas (solo una vez)
    if not new_state.get("habits_suggested"):
        suggested = _suggest_habits_from_symptoms(new_state.get("symptoms") or [])
        new_state["habits_suggested"] = suggested
        intro = (
            "Según los síntomas que has mencionado, estos hábitos son relevantes para contextualizar tu caso: "
            + ", ".join(suggested) + "."
        )
    else:
        suggested = new_state["habits_suggested"]
        intro = ""

    # 3) Asegurar entradas en 'habits'
    _ensure_habits_initialized(new_state, suggested)

    # 4) Recalcular pendientes y preguntas
    pending = _compute_pending_habit_slots(new_state)
    new_state["pending_habit_slots"] = pending
    new_state["missing_habits_fields"] = [f"{s['field']}: {s['habit_name']}" for s in pending]

    prev_text = (new_state.get("conversation_response") or "").strip()
    age = new_state.get("age")
    nationality = new_state.get("nationality")
    context = new_state.get("conversation_context")

    if not pending:
        # Nada pendiente: cierre amable
        pretty = _render_habits_text(age, nationality, context, intro, ["He registrado tus hábitos principales."])
        new_state["conversation_response"] = (prev_text + ("\n\n" if prev_text else "") + pretty).strip()
        return new_state

    # 5) Preguntar 1–2 pendientes
    to_ask = pending[:1]
    qs = [_question_for_habit_slot(s) for s in to_ask]
    new_state["last_asked_habit_slot"] = to_ask[0]

    pretty = _render_habits_text(age, nationality, context, intro, qs)
    new_state["conversation_response"] = (prev_text + ("\n\n" if prev_text else "") + pretty).strip()

    return new_state

# =========================
# Router
# =========================

def route_from_conversational(state):  # Literal["ask_user","to_symptom"]
    return "to_symptom" if state.get("conv_done") else "ask_user"

def route_from_symptom(state) -> Literal["ask_user", "to_habits"]:
    """
    Si ya no hay pendientes de síntomas (pending_slots vacío), pasamos a hábitos.
    De lo contrario, pedimos respuesta del usuario.
    """
    pending = state.get("pending_slots") or []
    return "to_habits" if len(pending) == 0 else "ask_user"

def route_from_habits(state) -> Literal["ask_user", "to_end"]:
    """
    Si ya no hay pendientes de hábitos (pending_habit_slots vacío), terminamos.
    Si faltan, pedimos respuesta del usuario.
    """
    pending_h = state.get("pending_habit_slots") or []
    return "to_end" if len(pending_h) == 0 else "ask_user"

# =========================
# Grafo minimal y runner
# =========================
def build_graph():
    g = StateGraph(AgentState)

    # Nodos que ya tienes implementados
    g.add_node("conversational", node_conversational)
    g.add_node("symptom", node_symptom_validator)
    g.add_node("habits", node_habits)
    
    g.set_entry_point("conversational")
    g.add_conditional_edges(
        "conversational",
        route_from_conversational,
        {
            "ask_user": END,
            "to_symptom": "symptom",
        },
    )
    g.add_conditional_edges(
        "symptom",
        route_from_symptom,
        {
            "ask_user": END,
            "to_habits": "habits",
        },
    )
    g.add_conditional_edges(
        "habits",
        route_from_habits,
        {
            "ask_user": END,
            "to_end": END,
        },
    )

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
    
