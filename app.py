# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# 1) Cargar variables de entorno (.env), igual que en main.py:contentReference[oaicite:3]{index=3}
load_dotenv()

from agent.graph import build_graph  # usamos el grafo directamente para mantener estado por turnos:contentReference[oaicite:4]{index=4}

st.set_page_config(page_title="🩺 Hackhaton TrivIAl centro médico", page_icon="🩺", layout="centered")

# --- Chequeo de API Key (evita ValueError en providers.py si falta):contentReference[oaicite:5]{index=5}---
if not os.getenv("COHERE_API_KEY"):
    st.error(
        "No se encontró **COHERE_API_KEY**. "
        "Crea un archivo `.env` en la raíz con `COHERE_API_KEY=tu_key` y reinicia `streamlit run app.py`."
    )
    st.stop()

# --- Inicialización de sesión ---
if "app" not in st.session_state:
    st.session_state.app = build_graph()  # compila el grafo una vez:contentReference[oaicite:6]{index=6}
if "state" not in st.session_state:
    st.session_state.state = None  # se crea cuando el usuario acepte consentimiento
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, text)]

# --- Sidebar: datos + consentimiento ---
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown(
        "Para continuar, por favor revisa y acepta el consentimiento.\n\n"
        "**Aviso**: Este sistema ofrece una *estimación probabilística*. "
        "No sustituye la valoración médica profesional."
    )
    consent = st.checkbox("Acepto el tratamiento de mis datos para continuar")
    age = st.number_input("Edad (opcional)", min_value=0, max_value=120, step=1, value=30)
    nationality = st.text_input("Nacionalidad (requerida para iniciar)", value="")
    start = st.button("Iniciar preconsulta")

st.title("🩺 Preconsulta con IA")
st.caption("Prototipo de anamnesis básica asistida. No es un diagnóstico médico.")

# --- Bloque de bienvenida / consentimiento (antes de iniciar) ---
if st.session_state.state is None:
    st.info(
        "Muy buenas tardes, bienvenido/a. Hoy le ayudaré con sus preocupaciones de salud mediante "
        "un **diagnóstico probabilístico**. Recuerde que esto es solo una **predicción** y no debe "
        "confiarse al 100%; debe apoyarse en profesionales de la salud para un veredicto certero.\n\n"
        "¿Tenemos su **consentimiento** para tratar sus datos y poder continuar?"
    )

    if start:
        if not consent:
            st.warning("Debes aceptar el consentimiento para continuar.")
        elif not (nationality and nationality.strip()):
            st.warning("La nacionalidad es requerida para iniciar.")
        else:
            # Crear estado inicial (equivalente al make_initial_state del main.py):contentReference[oaicite:7]{index=7}
            st.session_state.state = {
                "user_input": "",                 # el agente inicia (turno 0):contentReference[oaicite:8]{index=8}
                "age": int(age) if age is not None else None,
                "nationality": nationality.strip(),
                "reason_for_visit": None,
                "symptoms": [],
                "personal_history": None,
                "family_history": None,
                "missing_fields": [],
                "needs_another_symptom": False,
                "conversation_context": "",
                "conversation_response": "",
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

            # Turno 0: con user_input="" el conversacional abre y pregunta:contentReference[oaicite:9]{index=9}
            st.session_state.state = st.session_state.app.invoke(st.session_state.state)
            first_reply = (st.session_state.state.get("conversation_response") or "").strip() or "(sin mensaje)"
            st.session_state.history.append(("assistant", first_reply))

# --- Si ya hay estado, mostrar chat y manejar turnos ---
if st.session_state.state is not None:
    # Render histórico
    for role, msg in st.session_state.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(msg)

    # Deshabilitar input si ya terminó (output_done)
    disabled = bool(st.session_state.state.get("output_done"))

    user_msg = st.chat_input("Escribe tu mensaje...", disabled=disabled)

    if user_msg and not disabled:
        # 1) Mostrar turno del usuario
        st.session_state.history.append(("user", user_msg))

        # 2) Pasar el turno al grafo (manteniendo el mismo estado entre invocaciones):contentReference[oaicite:10]{index=10}
        st.session_state.state["user_input"] = user_msg
        st.session_state.state = st.session_state.app.invoke(st.session_state.state)

        # 3) Mostrar respuesta del agente
        bot_reply = (st.session_state.state.get("conversation_response") or "").strip() or "(sin mensaje)"
        st.session_state.history.append(("assistant", bot_reply))

        # Re-render inmediato
        st.rerun()

    # Si terminó el flujo, mostrar aviso
    if disabled:
        st.success("La sesión ha finalizado. Puedes recargar la página para iniciar una nueva conversación.")
