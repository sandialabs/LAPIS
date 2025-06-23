file(GLOB TESTFILES *.py)

set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${TESTFILES}
)

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

#add_test(NAME SPMV COMMAND python spmv.py)
add_subdirectory(Dialect)
#add_subdirectory(PartTensor)
#add_subdirectory(Transforms)
