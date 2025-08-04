# orphtools/config_loader.py
import yaml
import os

class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys, default=None):
        value = self.config
        try:
            for key in keys:
                value = value[key]
        except (KeyError, TypeError):
            return default
        return value

    def all(self):
        return self.config


# Example usage:
# config = ConfigLoader()
# max_len = config.get("model", "max_length")
# cleaned_path = config.get("data", "cleaned_dir")
