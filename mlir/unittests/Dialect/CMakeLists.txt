add_mlir_unittest(MLIRDialectTests
  BroadcastShapeTest.cpp
)
target_link_libraries(MLIRDialectTests
  PRIVATE
  MLIRIR
  MLIRDialect)

add_subdirectory(LLVMIR)
add_subdirectory(MemRef)
add_subdirectory(PartTensor)
add_subdirectory(SPIRV)
add_subdirectory(Transform)
add_subdirectory(Utils)
