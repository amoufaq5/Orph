# E:\Orph\src\utils\config.py
from __future__ import annotations
import os, yaml
from dotenv import load_dotenv

class Config:
    def __init__(self, path: str = "conf/config.yaml"):
        load_dotenv()  # .env overrides
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f) or {}

    def get(self, key: str, default=None):
        """
        Hierarchical get with .-separated keys; environment override:
        training.batch_size -> env TRAINING_BATCH_SIZE (if set)
        """
        env_key = key.upper().replace(".", "_")
        if env_key in os.environ:
            return _coerce(os.environ[env_key], default)
        node = self._cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default

    def require(self, key: str):
        v = self.get(key, None)
        if v is None:
            raise KeyError(f"Missing required config key: {key} (file: {self._path})")
        return v

def _coerce(val: str, default):
    # try to coerce env strings to type of default
    if default is None:
        return val
    t = type(default)
    if t is bool:
        return val.lower() in ("1","true","yes","on")
    if t is int:
        try: return int(val)
        except: return default
    if t is float:
        try: return float(val)
        except: return default
    return val
