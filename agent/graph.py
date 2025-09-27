# agent/graph.py
from typing import TypedDict, Literal, Optional, Dict, Any, Tuple
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate    
from .providers import chat_cohere
from typing import Dict, Any, List, Optional
import json, re

JUDGE_THRESHOLD = 0.40  

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

    judge_passes: int
    judge_route: Literal["to_output", "to_additional"]
    judge_decision: Literal["to_output", "to_additional"]
    
    additional_questions: List[Dict[str, Any]]     # [{"q": str, "answer": Optional[str], "asked": bool}]
    last_asked_additional: Optional[int]           # índice de la pregunta en curso
    
    # Señal del conversacional para ruteo
    conv_done: bool
    conv_route: Literal["ask_user", "to_symptom", "to_habits"]  # puedes usar sólo ask_user / to_habits si prefieres
    symptom_done: bool
    symptom_route: Literal["ask_user", "to_habits"]
    habits_done: bool
    habits_route: Literal["to_habits, to_classifier"]
    classifier_done: bool
    classifier_route: Literal["to_end", "to_judge"]
    additional_done: bool
    additional_route: Literal["ask_user", "to_classifier"]
    

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
{{
  "case_en": {{
    "age": number|null,
    "nationality": "string|null",
    "chief_complaint": "string|null",
    "symptoms": [
      {{
        "name": "string|null",
        "severity": "mild|moderate|severe|null",
        "frequency": "string|null",
        "onset": "string|null",
        "notes": "string|null"
      }}
    ],
    "personal_history": "string|null",
    "family_history": "string|null",
    "habits": [
      {{"name": "string", "regularity": "string|null"}}
    ]
  }},
  "differentials": [
    {{"condition": "string", "probability": 0.0, "rationale": "string"}},
    {{"condition": "string", "probability": 0.0, "rationale": "string"}},
    {{"condition": "string", "probability": 0.0, "rationale": "string"}}
  ]
}}

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

ADDITIONAL_SYSTEM = """Eres un asistente de salud. Recibes un caso clínico resumido y un top-3 de
diagnósticos diferenciales. Tu tarea: proponer EXACTAMENTE 3 preguntas cortas, claras y clínicas
(en español) que ayuden a diferenciar entre esas opciones. No des explicaciones, no menciones
diagnósticos en las preguntas, no sugieras tratamientos. Devuelve sólo una lista simple con viñetas."""

ADDITIONAL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", ADDITIONAL_SYSTEM),
    ("human",
     "CASO (resumen JSON en español): {case_es}\n\n"
     "DIFERENCIALES (máx 3): {diffs}\n\n"
     "Devuelve únicamente 3 líneas, cada una iniciando con '- ' y la pregunta.")
])

OUTPUT_SYSTEM = """Eres un asistente de salud para preconsulta.
Escribe en ESPAÑOL un mensaje breve, claro y empático para el/la paciente.
No diagnostiques ni indiques tratamientos específicos. No uses jerga técnica innecesaria.
Estructura:
1) Encabezado con la opción MÁS PROBABLE (condición) y su probabilidad en %.
   - Resume en 2 a 3 frases por qué podría encajar con el caso (usa el caso en inglés como referencia).
   - Indica pasos generales y seguros: autocuidado, señales de alarma y cuándo acudir a un profesional.
2) Otras posibilidades: lista con las otras 2 condiciones (solo nombre + 1 línea de contexto).
3) Mensaje ético final: deja claro que esto es apoyo con IA, que no reemplaza evaluación clínica,
   y que, si hay empeoramiento o señales de alarma, debe buscar atención.

No incluyas JSON ni tablas. Sé amable y directo/a.
"""

OUTPUT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", OUTPUT_SYSTEM),
    ("human",
     "CASE (EN JSON): {case_en}\n\n"
     "TOP1: {top1}\n"
     "TOP1_PROB: {top1_prob_percent}%\n"
     "TOP1_RATIONALE: {top1_rationale}\n\n"
     "OTHERS: {others}\n\n"
     "Redacta el mensaje final siguiendo la estructura indicada. "
     "No incluyas nada más que el texto para el/la paciente.")
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
      (o con _NO_INFO_TEXT si dice 'no sé' o no aporta).
    - Formula **1** pregunta pendiente y redacta con el prompt template.
    - Si ya no hay pendientes, no emite texto y deja listo el handoff al clasificador.
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
        # si el usuario no dijo nada útil, dejamos el slot para volver a preguntarlo más adelante

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

    # 4) Recalcular pendientes
    pending = _compute_pending_habit_slots(new_state)
    new_state["pending_habit_slots"] = pending
    new_state["missing_habits_fields"] = [f"{s['field']}: {s['habit_name']}" for s in pending]

    age = new_state.get("age")
    nationality = new_state.get("nationality")
    context = new_state.get("conversation_context")

    # 5) Si no hay pendientes: NO hablar y rutear al clasificador
    if not pending:
        new_state["habits_done"] = True
        new_state["habits_route"] = "to_classifier"
        # no escribir conversation_response para evitar dos voces;
        # el clasificador hablará en este mismo invoke
        return new_state

    # 6) Preguntar exactamente 1 pendiente
    slot = pending[0]
    q = _question_for_habit_slot(slot)
    new_state["last_asked_habit_slot"] = slot

    pretty = _render_habits_text(age, nationality, context, intro, [q])

    # **Sobrescribe** la respuesta (no concatenes texto previo)
    new_state["conversation_response"] = pretty

    # (opcional) Actualizar contexto compacto
    prev_ctx = new_state.get("conversation_context") or ""
    new_ctx = (prev_ctx + ("\n" if prev_ctx else "") + pretty).strip()
    if len(new_ctx) > 1400:
        new_ctx = new_ctx[-1400:]
    new_state["conversation_context"] = new_ctx

    return new_state

# =========================
# Classifier
# =========================

def node_classifier(state: "AgentState") -> "AgentState":
    """
    - Toma el estado actual, arma un JSON de caso (ES), y pide al LLM:
        * Traducción a EN (case_en)
        * Top-3 diferenciales con probabilidades
    - Actualiza state['case_en'], state['differentials'].
    - Prepara un resumen legible para el usuario (con disclaimer).
    - Marca classifier_done=True (router lo mandará al siguiente nodo/END).
    """
    new_state: AgentState = dict(state)

    # 1) Arma el caso (ES) y llama al LLM
    case_es = _build_patient_case_json_es(state)
    llm = chat_cohere()
    msgs = CLASSIFIER_TEMPLATE.format_messages(case_es=json.dumps(case_es, ensure_ascii=False))
    resp = llm.invoke(msgs).content

    # 2) Parseo tolerante
    data = _json_from_text(resp) or {}
    case_en = data.get("case_en") or {}
    diffs = data.get("differentials") or []

    # Normaliza diferenciales (top-3)
    cleaned_diffs = []
    for d in diffs[:3]:
        try:
            cleaned_diffs.append({
                "condition": str(d.get("condition", "")).strip(),
                "probability": float(d.get("probability", 0.0)),
                "rationale": str(d.get("rationale", "")).strip(),
            })
        except Exception:
            continue

    new_state["case_en"] = case_en
    new_state["differentials"] = cleaned_diffs
    new_state["classifier_done"] = True
    new_state["classifier_route"] = "to_end"  # cuando agregues 'judge', cambia a 'to_judge'

    # 3) Mensaje para el usuario (sin JSON, breve + disclaimer)
    if cleaned_diffs:
        lines = [f"- {d['condition']}: {round(d['probability']*100, 1)}% — {d['rationale']}" for d in cleaned_diffs]
        summary = "Posibles causas (no es diagnóstico):\n" + "\n".join(lines)
    else:
        summary = "He generado un resumen del caso. Aún no tengo un diferencial claro."

    disclaimer = "\n\nNota: Esto NO es un diagnóstico médico. Si presentas síntomas intensos o alarmantes, busca atención profesional."

    new_state["conversation_response"] = summary + disclaimer

    # (opcional) compacta algo de contexto
    prev_ctx = state.get("conversation_context") or ""
    new_ctx = (prev_ctx + ("\n" if prev_ctx else "") + summary).strip()
    if len(new_ctx) > 1400:
        new_ctx = new_ctx[-1400:]
    new_state["conversation_context"] = new_ctx

    return new_state


# =========================
# Judge
# =========================

def node_judge(state: "AgentState") -> "AgentState":
    """
    - Si alguna prob >= 0.80 => to_output
    - Si no, => to_additional
    - Conteo de vistas del juez; en la 3ra vez => to_output sí o sí.
    - NO emite texto (no toca conversation_response).
    """
    new_state: AgentState = dict(state)

    # Incrementa contador
    passes = int(new_state.get("judge_passes") or 0) + 1
    new_state["judge_passes"] = passes

    diffs = new_state.get("differentials") or []
    max_prob = 0.0
    for d in diffs:
        try:
            p = float(d.get("probability", 0.0))
        except Exception:
            p = 0.0
        if p > max_prob:
            max_prob = p

    if passes >= 2:
        decision = "to_output"
    elif max_prob >= JUDGE_THRESHOLD:
        decision = "to_output"
    else:
        decision = "to_additional"

    new_state["judge_decision"] = decision
    new_state["judge_route"] = decision
    # Importante: no tocar conversation_response ni context para evitar "dos voces"
    return new_state

# =========================
# Aditional
# =========================

def _render_additional_questions(case_es: dict, diffs: List[Dict[str, Any]]) -> List[str]:
    """Pide al LLM 3 preguntas discriminantes y las parsea como lista de strings."""
    llm = chat_cohere()
    difflist = ", ".join([str(d.get("condition","")).strip() for d in diffs[:3] if d.get("condition")])
    msgs = ADDITIONAL_TEMPLATE.format_messages(
        case_es=json.dumps(case_es, ensure_ascii=False),
        diffs=difflist or "(sin diferenciales)"
    )
    try:
        txt = llm.invoke(msgs).content
    except Exception:
        # Fallback muy simple si hay error/ratelimit
        return [
            "¿Has tenido fiebre recientemente?",
            "¿El dolor empeora al presionar o al moverte?",
            "¿Has notado cambios en tus deposiciones (sangre, diarrea o estreñimiento)?",
        ]
    # Parseo por líneas con '- '
    qs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        # limpieza menor
        line = re.sub(r"\s{2,}", " ", line)
        if line:
            qs.append(line)
        if len(qs) == 3:
            break
    # asegurar 3
    while len(qs) < 3:
        qs.append("¿Puedes aportar un detalle adicional que ayude a diferenciar la causa?")
    return qs

def _store_additional_as_symptom(state: "AgentState", question: str, answer: str):
    """Guarda la respuesta como un nuevo 'síntoma' con la respuesta en 'notes'."""
    symptoms = state.get("symptoms") or []
    entry = {
        "name": f"Pregunta adicional: {question[:60]}",
        "severity": None,
        "frequency": None,
        "onset": None,
        "notes": answer.strip() if answer else "No hay conocimiento sobre este aspecto",
    }
    symptoms.append(entry)
    state["symptoms"] = symptoms

def _is_no_info_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(x in t for x in ("no se","no sé","no recuerdo","n/a","no aplica","ninguno","ninguna"))

def node_additional(state: "AgentState") -> "AgentState":
    """
    - Si no existen 'additional_questions', genera 3 con LLM (según diferenciales y caso).
    - 1 pregunta por turno: si había una en curso, interpreta la respuesta y la guarda como 'síntoma'.
    - Si aún quedan preguntas sin responder, pregunta la siguiente y termina el turno.
    - Si todas están respondidas: no emite texto y rutea a 'classifier'.
    """
    new_state: AgentState = dict(state)
    user_text = state.get("user_input") or ""

    # 0) Si no hay lista de preguntas, generarla
    if not new_state.get("additional_questions"):
        case_es = _build_patient_case_json_es(state)
        diffs = state.get("differentials") or []
        qs = _render_additional_questions(case_es, diffs)
        new_state["additional_questions"] = [{"q": q, "answer": None, "asked": False} for q in qs]
        new_state["last_asked_additional"] = None

    # 1) Si había una pregunta en curso, registrar respuesta
    last_idx = new_state.get("last_asked_additional")
    if last_idx is not None:
        answer = None
        if _is_no_info_answer(user_text):
            answer = "No hay conocimiento sobre este aspecto"
        elif user_text.strip():
            answer = user_text.strip()
        # Si el usuario no dijo nada útil, dejamos el slot sin contestar y no avanzamos
        if answer is not None:
            new_state["additional_questions"][last_idx]["answer"] = answer
            new_state["additional_questions"][last_idx]["asked"] = True
            new_state["last_asked_additional"] = None
            # Guardar como síntoma
            _store_additional_as_symptom(new_state, new_state["additional_questions"][last_idx]["q"], answer)

    # 2) Buscar la siguiente pregunta pendiente (sin answer)
    next_idx = None
    for i, q in enumerate(new_state.get("additional_questions") or []):
        if not q.get("answer"):
            next_idx = i
            break

    # 3) Si ya completamos las 3 -> handoff a classifier (sin hablar)
    if next_idx is None:
        new_state["additional_done"] = True
        new_state["additional_route"] = "to_classifier"
        # No escribir conversation_response para evitar "doble voz"; el classifier hablará
        return new_state

    # 4) Hacer exactamente 1 pregunta (la siguiente pendiente)
    qtext = new_state["additional_questions"][next_idx]["q"]
    new_state["last_asked_additional"] = next_idx
    # Redacción final mínima (sin LLM extra)
    pretty = f"{qtext} Si no sabes o no aplica, puedes decir “no sé”."
    new_state["conversation_response"] = pretty

    # (opcional) Contexto compacto
    prev_ctx = new_state.get("conversation_context") or ""
    new_ctx = (prev_ctx + ("\n" if prev_ctx else "") + pretty).strip()
    if len(new_ctx) > 1400:
        new_ctx = new_ctx[-1400:]
    new_state["conversation_context"] = new_ctx

    return new_state

# =========================
# Output 
# =========================

def _render_output_text(case_en: dict, diffs: list) -> str:
    llm = chat_cohere()
    # preparar datos
    d0 = diffs[0]
    top1 = (d0.get("condition") or "").strip()
    p1 = float(d0.get("probability") or 0.0) * 100.0
    r1 = (d0.get("rationale") or "").strip()
    others = []
    for d in diffs[1:3]:
        name = (d.get("condition") or "").strip()
        if not name: 
            continue
        pr = float(d.get("probability") or 0.0) * 100.0
        rr = (d.get("rationale") or "").strip()
        others.append(f"{name} (~{round(pr,1)}%): {rr}")
    others_text = "; ".join(others) if others else "(sin alternativas)"

    msgs = OUTPUT_TEMPLATE.format_messages(
        case_en=json.dumps(case_en, ensure_ascii=False),
        top1=top1 or "(desconocido)",
        top1_prob_percent=round(p1, 1),
        top1_rationale=r1 or "(sin detalle)",
        others=others_text
    )
    try:
        return llm.invoke(msgs).content
    except Exception:
        # Fallback determinista mínimo
        lines = [f"Opción más probable: {top1 or 'Desconocida'} (≈{round(p1,1)}%)."]
        if r1:
            lines.append(f"Por qué podría encajar: {r1}")
        if others:
            lines.append("Otras posibilidades: " + "; ".join(o.split(":")[0] for o in others))
        lines.append(
            "Esto NO es un diagnóstico. Si presentas señales de alarma (dolor intenso, fiebre alta, vómitos persistentes, "
            "sangrado, dificultad para respirar, desmayo) o empeoras, busca atención profesional."
        )
        return "\n\n".join(lines)
    
# ==== Nodo: OutputConversation ====
def node_output(state: "AgentState") -> "AgentState":
    """
    - Toma case_en y differentials (top-3) y produce un mensaje final en español.
    - No pregunta nada. Sobrescribe conversation_response.
    - Señaliza que terminó.
    """
    new_state: AgentState = dict(state)

    diffs = new_state.get("differentials") or []
    if not diffs:
        new_state["conversation_response"] = (
            "Aún no tengo suficiente información para proponer posibilidades con confianza. "
            "Podemos revisar tus síntomas y hábitos nuevamente."
        )
        new_state["output_done"] = True
        return new_state

    # ordenar por prob (por si acaso)
    diffs_sorted = sorted(
        [d for d in diffs if d.get("condition")],
        key=lambda x: float(x.get("probability", 0.0)),
        reverse=True
    )
    case_en = new_state.get("case_en") or {}

    # Renderizar texto final (Cohere + fallback)
    text = _render_output_text(case_en, diffs_sorted)

    # Sobrescribir salida visible (no concatenar)
    new_state["conversation_response"] = text
    new_state["output_done"] = True

    # (opcional) actualizar contexto compacto
    prev_ctx = new_state.get("conversation_context") or ""
    new_ctx = (prev_ctx + ("\n" if prev_ctx else "") + text).strip()
    if len(new_ctx) > 1400:
        new_ctx = new_ctx[-1400:]
    new_state["conversation_context"] = new_ctx

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

def route_from_habits(state) -> Literal["ask_user", "to_classifier"]:
    pending_h = state.get("pending_habit_slots") or []
    return "to_classifier" if len(pending_h) == 0 else "ask_user"

def route_from_classifier(state) -> Literal["to_judge"]:
    # el clasificador siempre pasa por el juez
    return "to_judge"

def route_from_judge(state) -> Literal["to_output", "to_additional"]:
    # el nodo juez ya escribió la decisión en el estado
    return state.get("judge_route", "to_additional")

def route_from_additional(state) -> Literal["ask_user", "to_classifier"]:
    pending = [q for q in (state.get("additional_questions") or []) if not q.get("answer")]
    return "to_classifier" if len(pending) == 0 else "ask_user"


# =========================
# Grafo minimal y runner
# =========================
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("conversational", node_conversational)
    g.add_node("symptom", node_symptom_validator)
    g.add_node("habits", node_habits)
    g.add_node("classifier", node_classifier)
    g.add_node("judge", node_judge)  # ← nuevo
    g.add_node("additional", node_additional)
    g.add_node("output", node_output)
    
    g.set_entry_point("conversational")

    g.add_conditional_edges(
        "conversational",
        route_from_conversational,
        {"ask_user": END, "to_symptom": "symptom"},
    )
    g.add_conditional_edges(
        "symptom",
        route_from_symptom,
        {"ask_user": END, "to_habits": "habits"},
    )
    g.add_conditional_edges(
        "habits",
        route_from_habits,
        {"ask_user": END, "to_classifier": "classifier"},
    )

    # classifier -> judge
    g.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {"to_judge": "judge"},
    )

    
    g.add_conditional_edges(
    "judge",
    route_from_judge,
    {
        "to_output": "output",
        "to_additional": "additional",
    },
    )
    
    g.add_conditional_edges(
    "additional",
    route_from_additional,
    {
        "ask_user": END,
        "to_classifier": "classifier",
    },
    )
    
    g.add_edge("output", END)

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
    
