# main.py
from typing import Optional
from dotenv import load_dotenv
load_dotenv()  # COHERE_API_KEY, COHERE_MODEL, etc.

from agent.graph import build_graph, AgentState
import json
import sys

def pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def make_initial_state(age: Optional[int], nationality: Optional[str]) -> AgentState:
    return {
        "user_input": "",                 # el agente inicia
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
        # (si tus nodos usan estos, no hace daño inicializarlos vacíos)
        "pending_slots": [],
        "last_asked_slot": None,
        "asked_slots_history": [],
        "habits": [],
        "habits_suggested": [],
        "pending_habit_slots": [],
        "last_asked_habit_slot": None,
        "missing_habits_fields": [],
        "case_en": {},
        "differentials": [],
        "classifier_done": False,
        "judge_passes": 0,
        "additional_questions": [],
        "last_asked_additional": None,
        "additional_done": False,
        "output_done": False,
    }

def print_assistant(state: AgentState) -> None:
    msg = (state.get("conversation_response") or "").strip()
    if not msg:
        msg = "(sin mensaje)"
    print("\nAsistente:")
    print(msg)

def get_user_input() -> str:
    try:
        return input("\nTú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[Sesión cancelada por el usuario]")
        sys.exit(0)

if __name__ == "__main__":
    print("== LangGraph + Cohere (Agente de preconsulta) ==")

    try:
        age_str = input("Edad inicial (opcional): ").strip()
        age = int(age_str) if age_str else None
    except ValueError:
        age = None
    nationality = input("Nacionalidad inicial (opcional): ").strip() or None

    app = build_graph()
    state: AgentState = make_initial_state(age, nationality)

    # --- Turno 0: HABLA EL AGENTE ---
    state = app.invoke(state)
    print_assistant(state)

    # --- Bucle de una sola sesión: terminamos cuando Output da veredicto ---
    while True:
        # ¿Ya terminó con veredicto?
        if state.get("output_done"):
            print("\n[Fin de la sesión: se entregó el veredicto]")
            break

        # Solicitar respuesta del usuario
        user = get_user_input()
        if user.lower() in {"exit", "salir", "quit"}:
            print("\n[Sesión finalizada por el usuario]")
            break

        # Pasar la respuesta al grafo y continuar
        state["user_input"] = user
        state = app.invoke(state)
        print_assistant(state)

    # (Opcional) dump final resumido
    # print("\n[Resumen final]")
    # print(pretty({
    #     "reason_for_visit": state.get("reason_for_visit"),
    #     "symptoms": state.get("symptoms"),
    #     "personal_history": state.get("personal_history"),
    #     "family_history": state.get("family_history"),
    #     "differentials": state.get("differentials"),
    #     "output_done": state.get("output_done"),
    # }))
