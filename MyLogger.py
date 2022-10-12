import json
from logging import config, getLogger

class MyLogger:
    @staticmethod
    def initialize(logger_name:str, config_path: str):
        with open(config_path, 'r') as f:
            log_conf = json.load(f)
        config.dictConfig(log_conf)
        return getLogger(logger_name)
