add_subdirectory(backend)
add_definitions(-DTCNN_MIN_GPU_ARCH=$ENV{CUDA_ARCH} -DTCNN_SHAMPOO)

add_subdirectory(ext)

set ( KRR_GUIDED_PT
	render/guided/integrator.cpp
	render/guided/train.cu
)

set_source_files_properties (
	render/guided/integrator.cpp
	PROPERTIES LANGUAGE CUDA
)

set ( KRR_EXP_INCLUDE_ALL
	${KRR_INCLUDE_DIRS}
	${KRR_MISC_INCLUDE_ALL}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/ext/tcnn/include
	${CMAKE_CURRENT_SOURCE_DIR}/ext/tcnn/dependencies
)

message("KRR INCLUDE DIRS: ${KRR_EXP_INCLUDE_ALL}")

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/render/guided/guideditem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/guided/guideditem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/render/guided/guideditem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/guided/guideditem.soa)

set (KRR_EXP_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/render/guided/guideditem_soa.h
)

add_custom_target (krr_exp_soa_generated DEPENDS ${KRR_EXP_SOA_GENERATED})

include_directories (${KRR_EXP_INCLUDE_ALL})
include (${KRR_EXP_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
EXP_CUDA_COMPILE_EMBED(GUIDED_PTX render/guided/device.cu krr-guided krr_exp_soa_generated ${KRR_EXP_INCLUDE_ALL})
add_library(KRR_EXP_PTX STATIC
	${GUIDED_PTX}
)
add_dependencies(KRR_EXP_PTX krr_exp_soa_generated)

set ( KRR_EXP_LIBS_ALL
	krr_cuda_cfg
	krr_cuda_warning
	tiny-cuda-nn
	KRR_EXP_PTX
)

add_library (krr_exp_lib
	${KRR_GUIDED_PT}
	${KRR_EXP_SOA_GENERATED}
	${GUIDED_PTX}
)
add_dependencies(krr_exp_lib krr_exp_soa_generated)
set_property(TARGET krr_exp_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_link_libraries (krr_exp_lib PUBLIC ${KRR_EXP_LIBS_ALL})
target_include_directories(krr_exp_lib PUBLIC  ${KRR_EXP_INCLUDE_ALL})

add_executable( testbed main/main.cpp)
target_link_libraries (testbed PUBLIC ${KRR_EXP_LIBS_ALL})
target_link_libraries (testbed PUBLIC krr_exp_lib
	-WHOLEARCHIVE:$<TARGET_FILE:krr_exp_lib>)
target_link_libraries (testbed PUBLIC krr_lib
	-WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)

if (TARGET pykrr)
	message("Linking EXP library to python binding")
	TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_exp_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_exp_lib>)
endif()
	