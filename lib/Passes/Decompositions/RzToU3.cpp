
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir_utils.hpp"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_RZTOU3

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class CrzToCu3 final : public BaseMQSSPass<CrzToCu3>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrzToCu3)

  StringRef getArgument() const override { return "CrzToCu3"; }

  StringRef getDescription() const override {
    return "Decomposition pass that replaces Crz by Cu3";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rzOp = dyn_cast_or_null<quake::RzOp>(*op);
      if (!rzOp
          || rzOp.isAdj()
          || rzOp.getTargets().size() != 1
          || !rzOp.getControls().empty()
          || rzOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(rzOp->getContext());
      Value target = rzOp.getTargets()[0];
      Value control = rzOp.getControls()[0];
      Value param = rzOp.getParameters()[0];
      Location loc = rzOp.getLoc();
      auto constant_0 = mqss::support::quakeDialect::createFloatValue( rewriter, loc,0.0);
      
      rewriter.setInsertionPointAfter(rzOp);
      rewriter.create<quake::U3Op>(loc, ValueRange{constant_0, constant_0,param}, ValueRange{}, control,target);
      rewriter.eraseOp(rzOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createCrzToCu3Pass() {
  return std::make_unique<CrzToCu3>();
}