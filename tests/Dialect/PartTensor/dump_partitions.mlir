// RUN: %lapis-opt %s
// This is the example asked by Prof. Nasko as test for first part_tensor operation.
// This example is parsed without issue by lapis-opt (without any options.)

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
#relu_memory_access_map = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = max(X(i,j), 0)"
}
module {
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>) {
    %partition_plan = part_tensor.get_partitions %A: tensor<?x?xf32, #partEncoding> -> memref<?xindex>
    %c0 = arith.constant 0 : index
    %c2_i64 = arith.constant 2 : i64
    %c4_index = arith.constant 4 : index
    %n0 = memref.dim %partition_plan, %c0 : memref<?xindex>
    // might want to add assert saying %n0 is divisible by 4
    %num_points_index = part_tensor.get_num_partitions %A : tensor<?x?xf32, #partEncoding> -> index
    scf.forall(%i) in (%num_points_index) {
      %i_offset = arith.muli %i, %c4_index: index
      %v4 = vector.load %partition_plan[%i_offset]: memref<?xindex>, vector<4xindex>
      vector.print %v4 : vector<4xindex>
    }
    return
  }
}
