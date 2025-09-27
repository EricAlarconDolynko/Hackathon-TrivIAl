# lambdas/shared/app_runtime.py
from typing import Optional, Dict, Any
import time
from agent.graph import build_graph
from shared.logger import log

# Compila el grafo UNA sola vez por cold start
_APP = build_graph()

def _initial_state(age: Optional[int], nationality: Optional[str]) -> Dict[str, Any]:
    """Estado inicial compatible con tus nodos."""
    return {
        "user_input": "",                 # el agente abre
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

        # síntomas / validador
        "pending_slots": [],
        "last_asked_slot": None,
        "asked_slots_history": [],
        "symptom_done": False,

        # hábitos
        "habits": [],
        "habits_suggested": [],
        "pending_habit_slots": [],
        "last_asked_habit_slot": None,
        "missing_habits_fields": [],
        "habits_done": False,

        # clasificador / juez / additional / output
        "case_en": {},
        "differentials": [],
        "classifier_done": False,
        "judge_passes": 0,
        "additional_questions": [],
        "last_asked_additional": None,
        "additional_done": False,
        "output_done": False,

        # metadatos
        "_startedAt": int(time.time()),
    }

def start_session(age: Optional[int], nationality: Optional[str]) -> Dict[str, Any]:
    """Ejecuta el primer invoke del grafo (turno 0) y devuelve el estado resultante."""
    state = _initial_state(age, nationality)
    log.info("graph_invoke_start_session")
    out = _APP.invoke(state)
    return out

def run_turn(state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """Inyecta user_input y ejecuta EXACTAMENTE un invoke del grafo."""
    state = dict(state)
    state["user_input"] = user_input or ""
    log.info("graph_invoke_turn")
    out = _APP.invoke(state)
    return out
