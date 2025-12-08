#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_R1R1TOR1

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class R1R1ToR1 final : public BaseMQSSPass<R1R1ToR1>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(R1R1ToR1)

  StringRef getArgument() const override { return "R1R1ToR1"; }

  StringRef getDescription() const override {
    return "Collapse consecutive R1 gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto r1Op2 = dyn_cast_or_null<quake::R1Op>(*op);
      if (!r1Op2
          || r1Op2.getTargets().size() != 1
          || !r1Op2.getControls().empty()
          || r1Op2.getParameters().size() != 1) {
        return;
      }
      auto optional_rxOp1
          = getPreviousOperationOnTarget(r1Op2, r1Op2.getTargets()[0]);
      if (!optional_rxOp1) {
        return;
      }
      auto r1Op1 = dyn_cast_or_null<quake::R1Op>(*optional_rxOp1);
      if (!r1Op1
          || r1Op1.getTargets().size() != 1
          || !r1Op1.getControls().empty()
          || r1Op1.getParameters().size() != 1) {
        return;
      }
      auto r1Params = getOperationParameters(r1Op1);
      auto r1Params = getOperationParameters(r1Op2);
      if (r11Params.size() != 1 || r12Params.size() != 1) {
        return;
      }
      double angle = r11Params[0] + r12Params[0];
      IRRewriter rewriter(r1Op2->getContext());
      rewriter.setInsertionPointAfter(r1Op2);
      Location loc = r1Op1.getLoc();
      ValueRange targets = r1Op1.getTargets();
      Value params = createFloatValue(rewriter, loc, angle);
      rewriter.create<quake::R1Op>(loc, params, ValueRange{}, targets);
      rewriter.eraseOp(r1Op1);
      rewriter.eraseOp(r1Op2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createR1R1ToR1Pass() {
  return std::make_unique<R1R1ToR1>();
}