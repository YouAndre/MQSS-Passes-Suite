
#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_U3TORZRYRZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class U3ToRzRyRz final : public BaseMQSSPass<U3ToRzRyRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(U3ToRzRyRz)

  StringRef getArgument() const override { return "U3ToRzRyRz"; }

  StringRef getDescription() const override {
    return "Decompose U3 gate to Rz,Ry, Rz gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto u3Op2 = dyn_cast_or_null<quake::U3Op>(*op);
      if (!u3Op2
          || u3Op2.getTargets().size() != 1
          || !u3Op2.getControls().empty()
          || u3Op2.getParameters().size() != 2) {
        return;
      }
      
      auto params = u3Op2.getParameters();
      //auto u31Params = getOperationParameters(u3Op2);
      
      if (params.size() != 3) {
        return;
      }
      Value angle_0 = params[0];
      Value angle_1 = params[1];
      Value angle_2 = params[2];
      IRRewriter rewriter(u3Op2->getContext());
      rewriter.setInsertionPointAfter(u3Op2);
      Location loc = u3Op2.getLoc();
      ValueRange targets = u3Op2.getTargets();
      
      rewriter.create<quake::RzOp>(loc, false, ValueRange{angle_2},
                                      ValueRange{}, targets);
      rewriter.create<quake::RyOp>(loc, false, ValueRange{angle_0},
                                      ValueRange{}, targets);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{angle_1},
                                      ValueRange{}, targets);
      rewriter.eraseOp(u3Op2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createU3TORzRyRzPass() {
  return std::make_unique<U3ToRzRyRz>();
}