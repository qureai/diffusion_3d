import yaml


with open(r"default.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)
