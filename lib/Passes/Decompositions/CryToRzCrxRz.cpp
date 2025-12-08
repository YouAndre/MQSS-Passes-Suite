#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CRYTORZCRXRZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class CryToRzCrxRz final : public BaseMQSSPass<CryToRzCrxRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CryToRzCrxRz)

  StringRef getArgument() const override { return "CryToRzCrxRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Cry by Rz, Crx, and Rz";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cryOp = dyn_cast_or_null<quake::RyOp>(*op);
      if (!cryOp
          || cryOp.isAdj()
          || cryOp.getTargets().size() != 1
          || cryOp.getControls().size() != 1
          || cryOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(cryOp->getContext());
      Value control = cryOp.getControls()[0];
      Value target = cryOp.getTargets()[0];
      Value param = cryOp.getParameters()[0];
      Location loc = cryOp.getLoc();
      rewriter.setInsertionPointAfter(cryOp);
      auto constant_op_ry = mqss::support::quakeDialect::createFloatValue(rewriter,loc, -M_PI_2);
      auto constant_op_ry2 = mqss::support::quakeDialect::createFloatValue(rewriter,loc, M_PI_2);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{constant_op_ry},
                                        ValueRange{}, target);
      rewriter.create<quake::RxOp>(loc, false, param, control, target);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{constant_op_ry2},
                                        ValueRange{}, target);
      rewriter.eraseOp(cryOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCryToRzCrxRzPass() {
  return std::make_unique<CryToRzCrxRz>();
}