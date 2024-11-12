#include <queue>
#include <set>

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// KernelDAG definition 
//===----------------------------------------------------------------------===//

class KernelDAG {
  using Op = Operation;
  using OpSet = std::set<Op *>;
public:
  int size() { return graph.size(); }

  void addNode(Op *op) { graph[op] = SmallVector<Op *, 2>(); }

  void print() {
    for (auto nodeEdges : graph) {
      func::CallOp node = dyn_cast<func::CallOp>(nodeEdges.first);
      llvm::errs() << node.getCalleeAttr() << " -> {";
      for(auto edge : nodeEdges.second) {
        func::CallOp edgeCall = dyn_cast<func::CallOp>(edge);
        llvm::errs() << edgeCall.getCalleeAttr();
        if (edge != nodeEdges.second.back())
          llvm::errs() << ", ";
      }
      llvm::errs() << "}\n";
    }
  }

  // read this as "find an edge from op_a to op_b" 
  Op* find(Op *op_a, Op *op_b) {
    return *std::find(graph[op_a].begin(), graph[op_a].end(), op_b);
  }

  bool addEdge(Op *op_a, Op *op_b) {

    // edge already exists, so skip
    if (find(op_a, op_b) != *graph[op_a].end())
      return true;

    // possibly temporary push
    graph[op_a].push_back(op_b);

    // check for a cycle
    if (hasCycle()) {
      llvm::errs() << "Cycle found in kernel DAG\n";
      graph[op_a].pop_back(); // do not keep the push above
      return false;
    }

    // edge added
    return true;
  }

  bool hasCycle() { return !(topological_sort().size() == graph.size()); }

  SmallVector<Op *> topological_sort() {
    SmallVector<Op *> sorted;
    DenseMap<Op *, int> in_degrees;
    std::queue<Op *> S;

    // initialize number of incoming edges for each node
    for (auto entry : graph)
      in_degrees[entry.first] = 0;

    // compute in degrees for each node
    for (auto entry : graph) {
      for (auto val : graph[entry.first])
        in_degrees[val]++;
    }

    // form queue of all nodes with 0 incoming edges
    for (auto entry : in_degrees) {
      if (in_degrees[entry.first] == 0)
        S.push(entry.first);
    }

    // kahn's algorithm
    while (!S.empty()) {
      Op *from_node = S.front();
      S.pop();
      sorted.push_back(from_node);

      for (auto to_node : graph[from_node]) {
        in_degrees[to_node]--;

        if (in_degrees[to_node] == 0)
          S.push(to_node);
      }
    }

    return sorted;
  }
private:
  DenseMap<Op *, llvm::SmallVector<Op *, 2>> graph;
};



// TODO: test that this works for r/w dependences
KernelDAG createDependencyGraph(func::FuncOp mainFuncOp) {
  // a -> b <=> a depends on b <=> a must execute after b
  KernelDAG dependencyGraph;

  // find all other kernel calls that use the result of other calls
  for(func::CallOp call : mainFuncOp.getOps<func::CallOp>()) {
    dependencyGraph.addNode(call);
    for(Value result : call.getResults()) {
      for(Operation *user : result.getUsers()) {
        if (isa<func::CallOp>(user)) {
          if (!dependencyGraph.addEdge(user, call)) {
            llvm::errs() << "Cycle found... exiting\n";
            exit(-1);
          }
        }
      }
    }
  }

  return dependencyGraph;
}
