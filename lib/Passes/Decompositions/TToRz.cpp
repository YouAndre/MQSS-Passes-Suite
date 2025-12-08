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
class HToRzXRz final : public BaseMQSSPass<TToRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TToRz)

  StringRef getArgument() const override { return "TToRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass of T by Rz";
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
      if (!tOp) {
        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      
      OpBuilder mlirBuilder(kernel.getContext());
      mlirBuilder.setInsertionPoint(RzOp);
      Location loc = tOp.getLoc();
      auto target_qubit = tOp.getTargets();
      auto constant_op_rz = mqss::support::quakeDialect::createFloatValue(mlirBuilder,loc, M_PI_4);
      //createFloats((- M_PI), mlirBuilder, loc);
      mlirBuilder.create<quake::RyOp>(loc, false, ValueRange{constant_op_rz},
                                      ValueRange{}, target_qubit);
                                      
      IRRewriter rewriter(tOp->getContext());
      rewriter.eraseOp(tOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createTToRzPass() {
  return std::make_unique<TToRz>();
}
