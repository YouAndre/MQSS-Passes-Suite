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
#define GEN_PASS_DEF_STORZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;
using mlir::arith::ConstantOp;
namespace {
class HToRzXRz final : public BaseMQSSPass<SToRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SToRz)

  StringRef getArgument() const override { return "SToRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass of S by Rz";
  }

  auto createFloats(double value, OpBuilder &mlirBuilder, Location loc) {

      auto parameter =
          mlirBuilder.getFloatAttr(mlirBuilder.getF64Type(), value);
return mlirBuilder.create<ConstantOp>(loc, parameter);
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto sOp = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp) {
        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      
      OpBuilder mlirBuilder(kernel.getContext());
      mlirBuilder.setInsertionPoint(RzOp);
      Location loc = sOp.getLoc();
      auto target_qubit = sOp.getTargets();
      auto constant_op_rz = mqss::support::quakeDialect::createFloatValue(mlirBuilder,loc, M_PI_2);
      //createFloats((- M_PI), mlirBuilder, loc);
      mlirBuilder.create<quake::RyOp>(loc, false, ValueRange{constant_op_rz},
                                      ValueRange{}, target_qubit);
                                      
      IRRewriter rewriter(sOp->getContext());
      rewriter.eraseOp(sOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createSToRzPass() {
  return std::make_unique<SToRz>();
}
