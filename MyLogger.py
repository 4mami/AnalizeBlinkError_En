import json
from logging import getLogger, config

class MyLogger:
    @staticmethod
    def initialize(config_path: str):
        with open(config_path, 'r') as f:
            log_conf = json.load(f)
            config.dictConfig(log_conf)
            return getLogger(__name__)
