# lambdas/shared/response.py
import json
_DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}

def _resp(status: int, body: dict):
    return {
        "statusCode": status,
        "headers": _DEFAULT_HEADERS,
        "body": json.dumps(body, ensure_ascii=False),
    }

def ok(body: dict): return _resp(200, body)
def bad_request(body: dict): return _resp(400, body)
def server_error(body: dict): return _resp(500, body)
