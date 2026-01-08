# dependencies

find_package(Threads REQUIRED)

# flags

llama_add_compile_flags()

# tools

if (NOT EMSCRIPTEN)
    add_subdirectory(cli)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
endif()
