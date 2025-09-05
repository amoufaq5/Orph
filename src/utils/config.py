import yaml, os
from dataclasses import dataclass

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class OrphConfig:
    conf_root: str
    main: dict
    sources: dict
    routing: dict

def load_config(conf_dir: str = "conf"):
    main = load_yaml(os.path.join(conf_dir, "orph.yaml"))
    sources = load_yaml(os.path.join(conf_dir, "data_sources.yaml"))
    routing = load_yaml(os.path.join(conf_dir, "routing.yaml"))
    return OrphConfig(conf_root=conf_dir, main=main, sources=sources, routing=routing)
