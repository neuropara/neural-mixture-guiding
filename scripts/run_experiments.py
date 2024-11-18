import os
import sys
import json
import argparse
import multiprocessing

from common import *
import pykrr

test_scenes = [
	["common/configs/scenes/veach-ajar.json"],
	# ["common/configs/scenes/bathroom.json"],
	# ["common/configs/scenes/veach-bidir.json"]
]

def invoke_test(config: dict):
	import pykrr
	pykrr.run(config=config)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", type=str, default="guided", help="Method name")
	parser.add_argument("--budget_value", type=int, default=750, help="Budget value")
	args = parser.parse_args()

	for config_scene_file in test_scenes:
		config_method = json.load(open("common/configs/render/{}.json".format(args.method)))
		# load config
		config_method["passes"][0]["params"]["budget"]["value"] = args.budget_value

		scene_config = config_scene_file[0]
		config_scene = json.load(open(scene_config))
		method_name = os.path.splitext(os.path.basename(args.method))[0].replace("/", "-")
		scene_name = os.path.splitext(os.path.basename(scene_config))[0]
		print("Testing scene: {}".format(scene_name))	
		config_scene["global"] = {
			"reference": "common/configs/references/static/{}/reference_10b.exr".format(scene_name),
			"name": method_name
		}
		config_scene["output"] = "common/outputs/{}".format(scene_name)
		config_method.update(config_scene)
		p = multiprocessing.Process(target=invoke_test, args=(config_method,))
		p.start()
		p.join()
		print(">>>>>>>>>>>>>Done rendering reference image for: {}".format(scene_name))