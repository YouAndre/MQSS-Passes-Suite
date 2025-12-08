#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CYTOSCXSDG

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class CyToSCxSdg final : public BaseMQSSPass<CyToSCxSdg>,
                            public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CyToSCxSdg)

  StringRef getArgument() const override { return "CyToSCxSdg"; }

  StringRef getDescription() const override {
    return "Decomposition pass of two-qubits Cy by S, Cx, and Sdg";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cyOp = dyn_cast_or_null<quake::YOp>(*op);
      if (!cyOp || cyOp.getControls().size() != 1 ||
          cyOp.getTargets().size() != 1) {
        return;
      }

      IRRewriter rewriter(cyOp->getContext());
      Value control = cyOp.getControls()[0];
      Value target = cyOp.getTargets()[0];
      Location loc = cyOp.getLoc();
      rewriter.setInsertionPointAfter(cyOp);
      rewriter.create<quake::SOp>(loc, target);
      rewriter.create<quake::XOp>(loc, target, control);
      rewriter.create<quake::SOp>(loc, true,target);
      rewriter.eraseOp(cyOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCyToSCxSdgPass() {
  return std::make_unique<CyToSCxSdg>();
}