"""Read YAML config files.
"""
import yaml


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(yaml_str: str):
    """Load a YAML from method file."""
    cfg = yaml.safe_load(yaml_str)
    return cfg
