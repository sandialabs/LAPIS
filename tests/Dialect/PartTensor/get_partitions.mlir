// RUN: %lapis-opt %s -part-compiler
// This is the example asked by Prof. Nasko as test for first part_tensor operation.
// This example is parsed without issue by lapis-opt (without any options.)

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
module {
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>) -> memref<?xindex> {
    %partition_plan = part_tensor.get_partitions %A: tensor<?x?xf32, #partEncoding> -> memref<?xindex>
    return %partition_plan: memref<?xindex>
  }
}
