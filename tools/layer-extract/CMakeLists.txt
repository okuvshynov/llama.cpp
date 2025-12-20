set(TARGET llama-layer-extract)

add_executable(${TARGET} layer-extract.cpp)

target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
