# lambdas/handlers/session_turn_handler.py
import json
from shared.app_runtime import run_turn
from shared.state_store import get_state, put_state
from shared.response import ok, bad_request, server_error
from shared.logger import log

def handler(event, context):
    try:
        if not event.get("body"):
            return bad_request({"error": "Missing JSON body"})

        try:
            body = json.loads(event["body"])
        except Exception:
            return bad_request({"error": "Invalid JSON body"})

        session_id = (body.get("sessionId") or "").strip()
        user_input = (body.get("user_input") or "").strip()

        if not session_id:
            return bad_request({"error": "Missing sessionId"})
        # Permitimos user_input vacío (el grafo puede avanzar igual), pero puedes exigirlo si prefieres.

        # 1) Cargar estado
        state = get_state(session_id)
        if not state:
            return bad_request({"error": "Session not found"})

        # 2) Ejecutar un turno del grafo
        new_state = run_turn(state, user_input)

        # 3) Guardar
        put_state(session_id, new_state)

        # 4) Responder
        message = (new_state.get("conversation_response") or "").strip() \
                  or "¿Podrías confirmar o ampliar tu última respuesta?"
        done = bool(new_state.get("output_done"))

        log.info("session_turn_ok", sessionId=session_id, done=done)
        return ok({
            "sessionId": session_id,
            "message": message,
            "done": done
        })

    except Exception as e:
        log.exception("session_turn_error", error=str(e))
        return server_error({"error": "Internal error in /session/turn"})
