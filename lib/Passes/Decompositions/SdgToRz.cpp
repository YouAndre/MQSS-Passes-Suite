#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_SDGTORZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class SdgToRz final : public BaseMQSSPass<SdgToRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdgToRz)

  StringRef getArgument() const override { return "SdgToRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass that replaces a Sdg gate by Rz gates";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto sOp = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp
          || !sOp.isAdj()
          || sOp.getTargets().size() != 1
          || !sOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(sOp->getContext());
      Location loc = sOp.getLoc();
      auto target_qubit = sOp.getTargets();
      auto constant_op_rz = mqss::support::quakeDialect::createFloatValue(rewriter,loc, -M_PI_2);
      rewriter.setInsertionPointAfter(sOp);                            
      rewriter.create<quake::RzOp>(loc, false, constant_op_rz, ValueRange{}, target_qubit);
      rewriter.eraseOp(sOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createSdgToRzPass() {
  return std::make_unique<SdgToRz>();
}
