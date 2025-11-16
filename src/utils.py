import json
import yaml

def load_match_file(path):
    with open(path, "r", encoding="utf8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except:
        return yaml.safe_load(text)
