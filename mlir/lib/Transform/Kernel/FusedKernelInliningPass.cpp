/* Modified version of InlinerPass. Uses a custom profitability callback
 * function so that the inliner only operates on Operations that are tagged with
 * an "inline" attribute. Everything else, aside from some slight renaming, is
 * entirely the same as the built-in MLIR inliner.
 *
 * Exists so that we can still benefit from upstream MLIR changes without
 * worrying too much about conflicts.
 *
 * Ideally, a PR into upstream MLIR would be created that adds a stub in the
 * inliner that takes a profitability callback function on pass creation. This
 * way, many different custom inliners can be run over the same program.
*/

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/Passes.h"

#include "Transform/Kernel/KernelPasses.h"

namespace mlir {

static void defaultInlinerOptPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
}

#define GEN_PASS_DEF_INLINER
#include "mlir/Transforms/Passes.h.inc"

namespace kernel {

using PipelineTy = std::function<void(OpPassManager &)>;
using OpPipelinesTy = llvm::StringMap<OpPassManager>;

#define GEN_PASS_DEF_FUSEDKERNELINLININGPASS
#include "Transform/Kernel/KernelPasses.h.inc"

class FusedKernelInliningPass : public mlir::impl::InlinerBase<FusedKernelInliningPass> {
public:
  FusedKernelInliningPass();
  FusedKernelInliningPass(const FusedKernelInliningPass &) = default;
  FusedKernelInliningPass(PipelineTy defaultPipeline);
  FusedKernelInliningPass(PipelineTy defaultPipeline, OpPipelinesTy opPipelines);

  void runOnOperation() override;

  static LogicalResult runPipelineHelper(Pass &pass, OpPassManager &pipeline,
                                         Operation *op) {
    return mlir::cast<FusedKernelInliningPass>(pass).runPipeline(pipeline, op);
  }

private:
  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override;

  InlinerConfig config;
};

// constructor definitions
FusedKernelInliningPass::FusedKernelInliningPass()
    : FusedKernelInliningPass(defaultInlinerOptPipeline) {}
FusedKernelInliningPass::FusedKernelInliningPass(PipelineTy defaultPipelineArg)
    : FusedKernelInliningPass(std::move(defaultPipelineArg), OpPipelinesTy{}) {}
FusedKernelInliningPass::FusedKernelInliningPass(PipelineTy defaultPipeline,
                                     OpPipelinesTy opPipelines)
    : config(std::move(defaultPipeline), maxInliningIterations) {
  if (opPipelines.empty())
    return;

  for (auto &it : opPipelines)
    opPipelineList.addValue(it.second);
  config.setOpPipelines(std::move(opPipelines));
}

// adapted cost model function; only inline kernels that are tagged for inlining 
static bool isProfitableToInline(const Inliner::ResolvedCall &resolvedCall) {
  return resolvedCall.call->hasAttr("inline");
}

void FusedKernelInliningPass::runOnOperation() {
  CallGraph &cg = getAnalysis<CallGraph>();

  Operation *op = getOperation();
  if (!op->hasTrait<OpTrait::SymbolTable>()) {
    op->emitError() << " was scheduled to be run under the inliner, but does "
                    << "define a symbol table";
    return signalPassFailure();
  }

  auto profitabilityCb = [=](const Inliner::ResolvedCall &resolvedCall) {
    return isProfitableToInline(resolvedCall);
  };

  Inliner inliner(op, cg, *this, getAnalysisManager(), runPipelineHelper,
                  config, profitabilityCb);

  if(failed(inliner.doInlining()))
    return signalPassFailure();
}

LogicalResult FusedKernelInliningPass::initializeOptions(
    StringRef options,
    function_ref<LogicalResult(const Twine &)> errorHandler) {
  if(failed(Pass::initializeOptions(options, errorHandler)))
    return failure();

  if (!defaultPipelineStr.empty()) {
    std::string defaultPipelineCopy = defaultPipelineStr;
    config.setDefaultPipeline([=](OpPassManager &pm) {
      (void)parsePassPipeline(defaultPipelineCopy, pm);
    });
  }
  else if (defaultPipelineStr.getNumOccurrences()) {
    config.setDefaultPipeline(nullptr);
  }

  OpPipelinesTy pipelines;
  for (OpPassManager pipeline : opPipelineList)
    if (!pipeline.empty())
      pipelines.try_emplace(pipeline.getOpAnchorName(), pipeline);
  config.setOpPipelines(std::move(pipelines));

  config.setMaxInliningIterations(maxInliningIterations);

  return success();
}

std::unique_ptr<Pass> createFusedKernelInliningPass() {
  return std::make_unique<FusedKernelInliningPass>();
}

} // namespace kernel
} // namespace mlir

