# format as json and write to file
import json


def write_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
