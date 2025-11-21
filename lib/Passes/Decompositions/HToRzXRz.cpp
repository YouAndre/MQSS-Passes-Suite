#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_HTORZXRZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;
using mlir::arith::ConstantOp;
namespace {
class HToRzXRz final : public BaseMQSSPass<HToRzXRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HToRzXRz)

  StringRef getArgument() const override { return "HToRzXRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Rz by H, Rx, and H";
  }

  auto createFloats(double value, OpBuilder &mlirBuilder, Location loc) {

      auto parameter =
          mlirBuilder.getFloatAttr(mlirBuilder.getF64Type(), value);
return mlirBuilder.create<ConstantOp>(loc, parameter);
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp = dyn_cast_or_null<quake::HOp>(*op);
      if (!hOp) {
        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      
      OpBuilder mlirBuilder(kernel.getContext());
      mlirBuilder.setInsertionPoint(hOp);
      Location loc = hOp.getLoc();
      auto target_qubit = hOp.getTargets();
      auto constant_op_rz = createFloats(M_PI, mlirBuilder, loc);
      mlirBuilder.create<quake::RzOp>(loc, false, ValueRange{constant_op_rz},
                                      ValueRange{}, target_qubit);
                                      
      mlirBuilder.create<quake::XOp>(loc, false, ValueRange{},
                                      ValueRange{}, target_qubit);

      auto constant_op_rz_2 = createFloats(M_PI/2, mlirBuilder, loc);
      mlirBuilder.create<quake::RzOp>(loc, false, ValueRange{constant_op_rz_2},
                                      ValueRange{}, target_qubit);
      IRRewriter rewriter(hOp->getContext());
      rewriter.eraseOp(hOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createHToRzXRzPass() {
  return std::make_unique<HToRzXRz>();
}