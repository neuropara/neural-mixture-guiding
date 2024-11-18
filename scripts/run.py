import os
import sys
import json
import argparse

from krr import *
import pykrr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--render", type=str, default="common/configs/render/guided.json", help="Path to render configuration")
	parser.add_argument("--scene", type=str, default="common/configs/scenes/bathroom.json", help="Path to scene configuration")
	args = parser.parse_args()

	config = json.load(open(args.scene))
	config.update(json.load(open(args.render)))
	pykrr.run(config=config)