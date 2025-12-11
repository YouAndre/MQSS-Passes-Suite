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
#define GEN_PASS_DEF_TTORZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;
using mlir::arith::ConstantOp;
namespace {
class TdgToRz final : public BaseMQSSPass<TdgToRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TdgToRz)

  StringRef getArgument() const override { return "TdgToRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Tdg by Rz";
  }

  auto createFloats(double value, OpBuilder &mlirBuilder, Location loc) {

      auto parameter =
          mlirBuilder.getFloatAttr(mlirBuilder.getF64Type(), value);
return mlirBuilder.create<ConstantOp>(loc, parameter);
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto tOp = dyn_cast_or_null<quake::TOp>(*op);
      if (!tOp      
        || !tOp.isAdj()
        || tOp.getTargets().size() != 1
        || !tOp.getControls().empty()) {

        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      IRRewriter rewriter(tOp->getContext());
      rewriter.setInsertionPoint(tOp);
      Location loc = tOp.getLoc();
      auto target_qubit = tOp.getTargets();
      auto constant_op_rz = mqss::support::quakeDialect::createFloatValue(rewriter,loc,- M_PI_4);
      //createFloats((- M_PI), mlirBuilder, loc);
      rewriter.create<quake::RzOp>(loc, false, constant_op_rz, ValueRange{}, target_qubit);
      rewriter.eraseOp(tOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createTdgToRzPass() {
  return std::make_unique<TdgToRz>();
}
