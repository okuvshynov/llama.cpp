set(TARGET llama-dump-layer-data)
add_executable(${TARGET} dump-layer-data.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Need access to internal headers
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
