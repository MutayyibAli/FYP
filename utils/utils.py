import json


def load_config(type):
    with open(f"config/{type}.json", "r") as f:
        config = json.load(f)
    return config


def save_config(config, type):
    with open(f"config/{type}.json", "w") as f:
        json.dump(config, f)
