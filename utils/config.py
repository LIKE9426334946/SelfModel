import argparse
from pathlib import Path

import yaml


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="配置文件路径")
    parser.add_argument("--root", default=None, help="覆盖当前数据集根目录")
    parser.add_argument("--model", default=None, help="覆盖模型名称")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.root is not None:
        config["datasets"][config["dataset"]]["root"] = args.root
    if args.model is not None:
        config["model"]["name"] = args.model
    return config


def get_output_dir(config):
    return Path(config["output_root"]) / config["dataset"] / config["model"]["name"]


if __name__ == "__main__":
    config = load_config()
    get_output_dir(config)
    print("done")
