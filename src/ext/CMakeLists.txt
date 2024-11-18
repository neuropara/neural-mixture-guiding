
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(tcnn)
include_directories("tcnn/include")
include_directories("tcnn/dependencies")