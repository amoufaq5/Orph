from dotenv import load_dotenv
import os, yaml

class Config:
    def __init__(self, path="conf/config.yaml"):
        # Load .env first (overrides config.yaml)
        load_dotenv()
        with open(path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, key: str, default=None):
        # Check .env first
        env_val = os.getenv(key.upper().replace(".", "_"))
        if env_val is not None:
            return env_val
        # Fallback to YAML
        keys = key.split(".")
        val = self.cfg
        for k in keys:
            val = val.get(k, {})
        return val or default
