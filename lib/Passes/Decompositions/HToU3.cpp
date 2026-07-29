
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir_utils.hpp"
#include <cmath>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_HTOU3

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class HToU3 final : public BaseMQSSPass<HToU3>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HToU3)

  StringRef getArgument() const override { return "HToU3"; }

  StringRef getDescription() const override {
    return "Decomposition pass that replaces H by U3(pi/2, 0, pi)";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp = dyn_cast_or_null<quake::HOp>(*op);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }

      IRRewriter rewriter(hOp->getContext());
      Value target = hOp.getTargets()[0];
      Location loc = hOp.getLoc();

      rewriter.setInsertionPointAfter(hOp);
      auto halfPi = mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
      auto zero = mqss::support::quakeDialect::createFloatValue(rewriter, loc, 0.0);
      auto pi = mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI);

      rewriter.create<quake::U3Op>(loc, ValueRange{halfPi, zero, pi}, ValueRange{}, ValueRange{target});
      rewriter.eraseOp(hOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createHToU3Pass() {
  return std::make_unique<HToU3>();
}
