# lambdas/shared/state_store.py
import os
import time
import json
import boto3
from shared.logger import log

_TABLE = os.environ.get("DDB_TABLE", "medical-agent-sessions")
_TTL_HOURS = float(os.environ.get("SESSION_TTL_HOURS", "0") or 0)

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(_TABLE)

def put_state(session_id: str, state: dict) -> None:
    """Guarda el estado como JSON en DynamoDB (stateJson + updatedAt)."""
    ttl_attr = None
    if _TTL_HOURS and _TTL_HOURS > 0:
        ttl_attr = int(time.time() + _TTL_HOURS * 3600)

    item = {
        "sessionId": session_id,
        "stateJson": json.dumps(state, ensure_ascii=False),
        "updatedAt": int(time.time()),
    }
    if ttl_attr:
        item["ttl"] = ttl_attr

    _table.put_item(Item=item)
    log.info("ddb_put_state_ok", sessionId=session_id)

# Nota: el GET lo usarás en session_turn
def get_state(session_id: str) -> dict:
    res = _table.get_item(Key={"sessionId": session_id})
    item = res.get("Item") or {}
    if not item:
        return {}
    try:
        return json.loads(item.get("stateJson") or "{}")
    except Exception:
        log.error("ddb_get_state_json_parse_error", sessionId=session_id)
        return {}
