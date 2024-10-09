//===- KokkosOps.td - Kokkos dialect (base, enum attributes, ops) ----*- tablegen -*-===//

#ifndef KOKKOS_DIALECT_TD_H
#define KOKKOS_DIALECT_TD_H

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def Kokkos_Dialect : Dialect {
  let name = "kokkos";
  let cppNamespace = "::mlir::kokkos";
  let description = [{
     The `kokkos` dialect provides MLIR representations for Kokkos
     constructs like hierarchical parallelism, parallel_scan, Kokkos
     memory spaces and execution space instances.
  }];
}

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def Kokkos_MemorySpace : I64EnumAttr<
    "MemorySpace", "",
    [
      I64EnumAttrCase<"Host", 0, "host">,
      I64EnumAttrCase<"Device", 1, "device">,
      I64EnumAttrCase<"DualView", 2, "dualview">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_ExecutionSpace : I64EnumAttr<
    "ExecutionSpace", "",
    [
      I64EnumAttrCase<"Host", 0, "host">,
      I64EnumAttrCase<"Device", 1, "device">,
      I64EnumAttrCase<"TeamHandle", 2, "team_handle">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_ParallelLevel : I64EnumAttr<
    "ParallelLevel", "",
    [
      I64EnumAttrCase<"RangePolicy", 0, "range">,
      I64EnumAttrCase<"TeamVector", 1, "team_vector">,
      I64EnumAttrCase<"TeamThread", 2, "team_thread">,
      I64EnumAttrCase<"ThreadVector", 3, "thread_vector">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

def Kokkos_SingleLevel : I64EnumAttr<
    "SingleLevel", "",
    [
      I64EnumAttrCase<"PerTeam", 0, "per_team">,
      I64EnumAttrCase<"PerThread", 1, "per_thread">
    ]> {
  let cppNamespace = "::mlir::kokkos";
}

// Base op class.
class Kokkos_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kokkos_Dialect, mnemonic, traits>;

def RangeParallelOp : Kokkos_Op<"range_parallel",
    [AutomaticAllocationScope,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"kokkos::YieldOp">]> {
  let summary = "Kokkos base parallel for operation";
  let description = [{
  }];

  let arguments = (ins Kokkos_ExecutionSpace:$executionSpace,
                       Kokkos_ParallelLevel:$parallelLevel,
                       Variadic<Index>:$upperBound);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::kokkos::ExecutionSpace":$executionSpace, "::mlir::kokkos::ParallelLevel":$parallelLevel,
    "ValueRange":$upperBounds, "TypeRange":$resultTypes)>
  ];

  let extraClassDeclaration = [{
    Region& getLoopBody() { return getRegion(); }
    ValueRange getInductionVars() {
      return getBody()->getArguments();
    }
    unsigned getNumLoops() { return getUpperBound().size(); }
    unsigned getNumReductions() { return getNumResults(); }

    // Get the reduction op for this parallel, if any. It may be directly nested or inside a single.
    // NOTE: in future LLVM, scf.parallel can support multiple reductions but there will still be only one
    // scf.reduce op per loop, which updates them all. kokkos.update_reduction will work the same way.
    kokkos::UpdateReductionOp getReduction();
  }];

  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TeamParallelOp : Kokkos_Op<"team_parallel",
    [AutomaticAllocationScope,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"kokkos::YieldOp">]> {
  let summary = "Kokkos TeamPolicy parallel for/reduce";

  let arguments = (ins Index:$leagueSize,
                       Index:$teamSizeHint,
                       Index:$vectorLengthHint);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;

  // ValueRange (body block arguments) should contain exactly 4 values:
  // league size, team size, league rank, team rank

  // Because we don't know the hardware architecture ahead of time,
  // the team size and vector length passed into the body are just hints.
  // The real team size/vector length have to be computed at runtime.
  let builders = [
    OpBuilder<(ins "Value":$leagueSize, "Value":$teamSizeHint, "Value":$vectorLengthHint,
      "TypeRange":$resultTypes)>
  ];

  let extraClassDeclaration = [{
    Region& getLoopBody() { return getRegion(); }
    unsigned getNumReductions() { return getNumResults(); }
    // leagueSize is an operand, so getLeagueSize() already exists.
    Value getTeamSize() { return getLoopBody().front().getArguments()[1]; }
    Value getLeagueRank() { return getLoopBody().front().getArguments()[2]; }
    Value getTeamRank() { return getLoopBody().front().getArguments()[3]; }

    // Get the reduction op for this parallel, if any. It may be directly nested or inside a single.
    kokkos::UpdateReductionOp getReduction();
  }];

  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def ThreadParallelOp : Kokkos_Op<"thread_parallel",
    [AutomaticAllocationScope,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"kokkos::YieldOp">]> {
  let summary = "Kokkos 2-level parallel for/reduce";
  let description = [{
    A `kokkos.thread_parallel` op represents a 2-level nested parallel loop.
    Each outer level iteration is mapped to a Kokkos thread using a combination of `TeamPolicy` and `TeamThreadRange`.
    Each inner level iteration is mapped to a vector lane within the thread using `ThreadVectorRange`.
    The team size and distribution of outer iterations to teams is handled automatically by the
    Kokkos emitter and at runtime.
    Because the outer iterations must be independent and Kokkos ThreadVector loops have
    implicit synchronization, `kokkos.team_barrier` is never used inside this op. However,
    per-thread `kokkos.single` can be used to ensure op(s) with side effects happen exactly once
    per outer iteration.
  }];

  let arguments = (ins Index:$numIters,
                       Index:$vectorLengthHint);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Value":$numIters, "Value":$vectorLengthHint,
    "TypeRange":$resultTypes)>
  ];

  let extraClassDeclaration = [{
    Region& getLoopBody() { return getRegion(); }
    unsigned getNumReductions() { return getNumResults(); }
    Value getInductionVar() { return getLoopBody().front().getArguments()[0]; }

    // Get the reduction op for this parallel, if any. It may be directly nested or inside a single.
    kokkos::UpdateReductionOp getReduction();
  }];

  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def TeamBarrierOp : Kokkos_Op<"team_barrier", [HasParent<"TeamParallelOp">]> {
  let summary = "Synchronize all threads in a team. In C++, t.team_barrier()";
  let description = [{
    Kokkos team barrier, to be called within a TeamPolicy context. All threads in
    the team must reach the barrier together to avoid deadlock, but this is enforced by the
    fact that TeamParallelOp must be the parent.
    ```
    kokkos.team_barrier
    ```
  }];
}

def SyncOp : Kokkos_Op<"sync"> {

  let description = [{
    For a memref with DualView memory space, sync its contents to the
    provided memory space if it has been marked modified in the opposite space.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to sync">:$view,
                       Kokkos_MemorySpace:$memorySpace);

  let assemblyFormat = [{
    $view attr-dict `:` type($view)
  }];

  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def ModifyOp : Kokkos_Op<"modify"> {
  let description = [{
    For a memref with DualView memory space, mark its contents as modified
    in the given space. This operation is idempotent.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to mark modified">:$view,
                       Kokkos_MemorySpace:$memorySpace);

  let assemblyFormat = [{
    $view attr-dict `:` type($view)
  }];

  let hasCanonicalizer = 0;
  let hasFolder = 0;
}

def SingleOp : Kokkos_Op<"single", [SingleBlockImplicitTerminator<"kokkos::YieldOp">, RecursiveMemoryEffects, NoRegionArguments, InferTypeOpAdaptor]> {
  let summary = "Restricts a block to execute exactly once per team or per thread";
  let description = [{
    "Restricts a block to execute exactly once per team or per thread"
  }];

  let arguments = (ins Kokkos_SingleLevel:$level);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def UpdateReductionOp : Kokkos_Op<"update_reduction",
[ParentOneOf<["RangeParallelOp", "TeamParallelOp", "ThreadParallelOp", "SingleOp"]>]>
{
  let summary = "Contribute to a reduction inside a parallel loop";
  let description = [{
    Update a partial reduction inside a parallel loop.
    The ``update`` operand is the value to contribute to the reduction.
    The body defines the ``join`` operation for combining two partial reductions
    or a partial reduction and a value.
    The ``identity`` operand is the value for which ``join(x, identity) == x`` for all ``x``.
    The result of the reduce operation's body must have the same type as the
    operands and associated result value of the parallel loop operation.
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$update, "Value":$identity,
      CArg<"function_ref<void (OpBuilder &, Location, Value, Value)>",
           "nullptr">:$bodyBuilderFn)>
  ];

  let arguments = (ins AnyType:$update, AnyType:$identity);
  let hasCustomAssemblyFormat = 0;
  let regions = (region SizedRegion<1>:$reductionOperator);
  let hasRegionVerifier = 0;
}

def YieldOp : Kokkos_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["RangeParallelOp", "TeamParallelOp", "ThreadParallelOp", "SingleOp", "UpdateReductionOp"]>]> {
  let summary = "Region terminator";
  let description = [{ 
    "kokkos.yield" terminates the region of a parallel, single or reduction op.
    If enclosed in a parallel op, it should have zero operands.
    If enclosed in a single, the operand types must match the results
    of the single, and these are the values that will be broadcast by the single.
    If enclosed in the i'th region of a reduce, the operand is the i'th updated
    partial reduction.
    If the enclosing operation has no results and a yield is not added explicitly,
    then the custom IR builders will insert one automatically.
  }];
    
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  
  let assemblyFormat = 
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif
