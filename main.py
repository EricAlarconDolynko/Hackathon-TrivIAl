# main.py
from typing import Optional
from dotenv import load_dotenv
load_dotenv()  # COHERE_API_KEY, etc.
from agent.graph import build_graph, AgentState
import json

def pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def make_initial_state(age: Optional[int], nationality: Optional[str]) -> AgentState:
    return {
        "user_input": "",                 # <-- vacío: el agente inicia
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

if __name__ == "__main__":
    print("== LangGraph + Cohere (Agente de preconsulta) ==")
    print("Escribe 'exit' para salir.\n")

    # (Opcional) semilla demográfica
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
    print("\nAsistente:")
    print(state.get("conversation_response", ""))
    summary = {
        "reason_for_visit": state.get("reason_for_visit"),
        "symptoms": state.get("symptoms"),
        "personal_history": state.get("personal_history"),
        "family_history": state.get("family_history"),
        "missing_fields": state.get("missing_fields"),
        "needs_another_symptom": state.get("needs_another_symptom"),
    }
    print("\n[Estado resumido]")
    print(pretty(summary))

    # --- Loop de conversación (ahora hablas tú) ---
    while True:
        user = input("\nTú: ").strip()
        if user.lower() in {"exit", "salir", "quit"}:
            break

        state["user_input"] = user
        state = app.invoke(state)

        print("\nAsistente:")
        print(state.get("conversation_response", ""))

        summary = {
            "reason_for_visit": state.get("reason_for_visit"),
            "symptoms": state.get("symptoms"),
            "personal_history": state.get("personal_history"),
            "family_history": state.get("family_history"),
            "missing_fields": state.get("missing_fields"),
            "needs_another_symptom": state.get("needs_another_symptom"),
        }
        #print("\n[Estado resumido]")
        #print(pretty(summary))
