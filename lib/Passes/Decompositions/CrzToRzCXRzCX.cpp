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
#define GEN_PASS_DEF_CRZTORZCXRZCX

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class CrzToHCrxH final : public BaseMQSSPass<CrzToHCrxH>,
                         public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrzToHCrxH)

  StringRef getArgument() const override { return "CrzToHCrxH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Crz by H, Crx, and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto crzOp = dyn_cast_or_null<quake::RzOp>(*op);
      if (!crzOp || crzOp.isAdj() || crzOp.getTargets().size() != 1 ||
          crzOp.getControls().size() != 1 ||
          crzOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(crzOp->getContext());
      Value control = crzOp.getControls()[0];
      Value target = crzOp.getTargets()[0];
      Value param = crzOp.getParameters()[0];
      Location loc = crzOp.getLoc();

      std::vector<double> params = mqss::support::quakeDialect::getOperationParameters(crzOp);
      double angle = params[0];
      // double angle_1 = params[1];

      auto constant_float_2 = mqss::support::quakeDialect::createFloatValue(
          rewriter, loc, angle / 2);
      auto constant_float__2 = mqss::support::quakeDialect::createFloatValue(
          rewriter, loc, angle / -2);

      rewriter.create<quake::RzOp>(loc, false, ValueRange{constant_float_2},
                                   ValueRange{}, target);
      rewriter.setInsertionPointAfter(crzOp);
      rewriter.create<quake::XOp>(loc, control, target);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{constant_float__2},
                                   ValueRange{}, target);
      rewriter.create<quake::XOp>(loc, control, target);
      rewriter.eraseOp(crzOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCrzToHCrxHPass() {
  return std::make_unique<CrzToHCrxH>();
}