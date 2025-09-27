# lambdas/shared/logger.py
import logging, json

class _Log:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self._l = logging.getLogger("agent")

    def info(self, msg, **kwargs):
        self._l.info("%s %s", msg, json.dumps(kwargs, ensure_ascii=False))

    def error(self, msg, **kwargs):
        self._l.error("%s %s", msg, json.dumps(kwargs, ensure_ascii=False))

    def exception(self, msg, **kwargs):
        self._l.exception("%s %s", msg, json.dumps(kwargs, ensure_ascii=False))

log = _Log()
