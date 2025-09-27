# lambdas/handlers/session_start_handler.py
import json
import uuid
from shared.app_runtime import start_session
from shared.state_store import put_state
from shared.response import ok, bad_request, server_error
from shared.logger import log

def handler(event, context):
    try:
        body = {}
        if event.get("body"):
            try:
                body = json.loads(event["body"])
            except Exception:
                return bad_request({"error": "Invalid JSON body"})

        age = body.get("age")
        nationality = body.get("nationality")

        # 1) Ejecutar el turno 0 (habla el agente)
        state = start_session(age=age, nationality=nationality)

        # 2) Crear sessionId y guardar estado
        session_id = str(uuid.uuid4())
        put_state(session_id, state)

        # 3) Responder al frontend
        message = (state.get("conversation_response") or "").strip() or "Hola, ¿en qué puedo ayudarte hoy?"
        done = bool(state.get("output_done"))

        log.info("session_start_ok", sessionId=session_id, done=done)
        return ok({
            "sessionId": session_id,
            "message": message,
            "done": done
        })

    except Exception as e:
        log.exception("session_start_error", error=str(e))
        return server_error({"error": "Internal error in /session/start"})
