cmake_minimum_required(VERSION 3.18)
project(mps_bench_native LANGUAGES CXX CUDA)

# sm_80 is stated explicitly: A30 is Ampere (compute capability 8.0). We do NOT
# rely on the build machine auto-detecting a GPU architecture, because the image
# is built on hosts that may have no GPU at all.
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# IMPORTANT: device assert (F8) must remain enabled. Release adds -DNDEBUG by
# default, which would compile away `assert` in device code and make the F8 case
# silently non-functional. We strip NDEBUG from every configuration.
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

add_executable(device_probe device_probe.cu)
add_executable(memory_probe memory_probe.cu)
add_executable(fault_injector fault_injector.cu)

foreach(target device_probe memory_probe fault_injector)
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(${target} PRIVATE cuda)
endforeach()

install(TARGETS device_probe memory_probe fault_injector
        RUNTIME DESTINATION bin)
