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
#define GEN_PASS_DEF_YTORY

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;
using mlir::arith::ConstantOp;
namespace {
class YToRy final : public BaseMQSSPass<YToRy>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YToRy)

  StringRef getArgument() const override { return "XToRx"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Y by Ry";
  }

  auto createFloats(double value, OpBuilder &mlirBuilder, Location loc) {

      auto parameter =
          mlirBuilder.getFloatAttr(mlirBuilder.getF64Type(), value);
return mlirBuilder.create<ConstantOp>(loc, parameter);
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto yOp = dyn_cast_or_null<quake::YOp>(*op);
      if (!yOp) {
        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      IRRewriter rewriter(yOp->getContext());

      rewriter.setInsertionPoint(yOp);
      Location loc = yOp.getLoc();
      auto target_qubit = yOp.getTargets();
      auto constant_op_ry = mqss::support::quakeDialect::createFloatValue(rewriter,loc, -M_PI);
      //createFloats((- M_PI), mlirBuilder, loc);
      rewriter.create<quake::RyOp>(loc, false, ValueRange{constant_op_ry},
                                      ValueRange{}, target_qubit);
      rewriter.eraseOp(yOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createYToRyPass() {
  return std::make_unique<YToRy>();
}
