#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/IR/ValueRange.h>

namespace mqss::opt {
#define GEN_PASS_DEF_U2U2TOU2

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class U2U2ToU2 final : public BaseMQSSPass<U2U2ToU2>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(U2U2ToU2)

  StringRef getArgument() const override { return "U2U2ToU2"; }

  StringRef getDescription() const override {
    return "Collapse consecutive U2 gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto u2Op2 = dyn_cast_or_null<quake::U2Op>(*op);
      if (!u2Op2
          || u2Op2.getTargets().size() != 1
          || !u2Op2.getControls().empty()
          || u2Op2.getParameters().size() != 2) {
        return;
      }
      auto optional_u2Op1
          = getPreviousOperationOnTarget(u2Op2, u2Op2.getTargets()[0]);
      if (!optional_u2Op1) {
        return;
      }
      auto u2Op1 = dyn_cast_or_null<quake::U2Op>(*optional_u2Op1);
      if (!u2Op1
          || u2Op1.getTargets().size() != 1
          || !u2Op1.getControls().empty()
          || u2Op1.getParameters().size() != 2) {
        return;
      }
      auto u21Params = getOperationParameters(u2Op1);
      auto u22Params = getOperationParameters(u2Op2);
      if (u21Params.size() != 2 || u22Params.size() != 2) {
        return;
      }
      double angle_0 = u21Params[0] + u22Params[0];
      double angle_1 = u21Params[1] + u22Params[1];
      IRRewriter rewriter(u2Op2->getContext());
      rewriter.setInsertionPointAfter(u2Op2);
      Location loc = u2Op1.getLoc();
      ValueRange targets = u2Op1.getTargets();
      Value param_0 = createFloatValue(rewriter, loc, angle_0);
      Value param_1 = createFloatValue(rewriter, loc, angle_1);
      rewriter.create<quake::U2Op>(loc, ValueRange{param_0, param_1}, ValueRange{}, targets);

      rewriter.eraseOp(u2Op1);
      rewriter.eraseOp(u2Op2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createU2U2ToU2Pass() {
  return std::make_unique<U2U2ToU2>();
}