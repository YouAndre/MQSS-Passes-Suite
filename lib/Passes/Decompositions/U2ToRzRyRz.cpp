#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_U3ToRzRyRz

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"
}

using namespace mlir;

namespace {
class U2ToRzRyRz final : public BaseMQSSPass<U2ToRzRyRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(U2ToRzRyRz)

  StringRef getArgument() const override { return "U2ToRzRyRz"; }

  StringRef getDescription() const override {
    return "Decompose U2 gate to Rz,Ry, Rz gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto u2Op = dyn_cast_or_null<quake::U2Op>(*op);
      if (!u2Op
          || u2Op.getTargets().size() != 1
          || !u2Op.getControls().empty()
          || u2Op.getParameters().size() != 2) {
        return;
      }
      
      auto params = u2Op.getParameters();
      //auto u31Params = getOperationParameters(u2Op);
      
      if (params.size() != 2) {
        return;
      }
      Value angle_0 = params[0];
      Value angle_1 = params[1];

      IRRewriter rewriter(u2Op->getContext());
      rewriter.setInsertionPointAfter(u2Op);
      Location loc = u2Op.getLoc();
      ValueRange targets = u2Op.getTargets();

      auto constant_0 = mqss::support::quakeDialect::createFloatValue( rewriter, loc,M_PI_2);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{angle_0},
                                      ValueRange{}, targets);
      rewriter.create<quake::RyOp>(loc, false, ValueRange{constant_0},
                                      ValueRange{}, targets);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{angle_1},
                                      ValueRange{}, targets);
      rewriter.eraseOp(u2Op);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createU2ToRzRyRzPass() {
  return std::make_unique<U2ToRzRyRz>();
}