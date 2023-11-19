cmake_minimum_required(VERSION 3.27)
project(alpacker)

set(CMAKE_CXX_STANDARD 17)

add_executable(alpacker
        main.cpp)
target_link_libraries(alpacker PRIVATE dpp common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(alpacker PRIVATE include)
