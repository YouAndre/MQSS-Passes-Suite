#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_U3U3TOU3

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class U3U3ToU3 final : public BaseMQSSPass<U3U3ToU3>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(U3U3ToU3)

  StringRef getArgument() const override { return "U3U3ToU3"; }

  StringRef getDescription() const override {
    return "Collapse consecutive U3 gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto u3Op2 = dyn_cast_or_null<quake::U3Op>(*op);
      if (!u3Op2
          || u3Op2.getTargets().size() != 1
          || !u3Op2.getControls().empty()
          || u3Op2.getParameters().size() != 3) {
        return;
      }
      auto optional_u3Op1
          = getPreviousOperationOnTarget(u3Op2, u3Op2.getTargets()[0]);
      if (!optional_u3Op1) {
        return;
      }
      auto u3Op1 = dyn_cast_or_null<quake::U3Op>(*optional_u3Op1);
      if (!u3Op1
          || u3Op1.getTargets().size() != 1
          || !u3Op1.getControls().empty()
          || u3Op1.getParameters().size() != 3) {
        return;
      }
      auto u31Params = getOperationParameters(u3Op1);
      auto u32Params = getOperationParameters(u3Op2);
      if (u31Params.size() != 3 || u32Params.size() != 3) {
        return;
      }
      double angle_0 = u31Params[0] + u32Params[0];
      double angle_1 = u31Params[1] + u32Params[1];
      double angle_2 = u31Params[2] + u32Params[2];
      IRRewriter rewriter(u3Op2->getContext());
      rewriter.setInsertionPointAfter(u3Op2);
      Location loc = u3Op1.getLoc();
      ValueRange targets = u3Op1.getTargets();
      Value param_0 = createFloatValue(rewriter, loc, angle_0);
      Value param_1 = createFloatValue(rewriter, loc, angle_1);
      Value param_2 = createFloatValue(rewriter, loc, angle_2);
      rewriter.create<quake::U3Op>(loc, ValueRange{param_0, param_1,param_2}, ValueRange{}, targets);

      rewriter.eraseOp(u3Op1);
      rewriter.eraseOp(u3Op2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createU3U3ToU3Pass() {
  return std::make_unique<U3U3ToU3>();
}