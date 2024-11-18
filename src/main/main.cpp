#include <iostream>

#include "file.h"
#include "logger.h"

#include "main/renderer.h"
#include "scene/importer.h"

#include "json.hpp"

KRR_NAMESPACE_BEGIN

extern "C" int main(int argc, char *argv[]) {
	gpContext = Context::SharedPtr(new Context());
	fs::current_path(File::cwd());
	logInfo("Working directory: " + string(KRR_PROJECT_DIR));

	string sceneConfig = "common/configs/scenes/veach-ajar.json";
	string renderConfig = "common/configs/render/guided.json";
	string globalConfig = "common/configs/mutable.json";

	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-scene") 
			sceneConfig = string(argv[++i]);
		else if (string(argv[i]) == "-method") 
			renderConfig = string(argv[++i]);
	}

	RenderApp app;	
	app.loadConfigFrom(globalConfig);
	app.loadConfigFrom(sceneConfig);
	app.loadConfigFrom(renderConfig);
	
	if (!File::outputDir().empty())
		File::setOutputDir(File::resolve("common/outputs") 
			/ fs::path(sceneConfig).stem() / fs::path(renderConfig).stem());

	app.run();
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END