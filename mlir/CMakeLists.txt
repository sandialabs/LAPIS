set(LAPIS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories( "include")
include_directories( ${LAPIS_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
