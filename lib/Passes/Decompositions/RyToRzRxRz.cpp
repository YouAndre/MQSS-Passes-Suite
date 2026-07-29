#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_RYTORZRXRZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;
using mlir::arith::ConstantOp;
namespace {
class RyToRzRxRz final : public BaseMQSSPass<RyToRzRxRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RyToRzRxRz)

  StringRef getArgument() const override { return "RyToRzRxRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Ry by Rz, Rx, and Rz";
  }

  auto createFloats(double value, OpBuilder &mlirBuilder, Location loc) {

      auto parameter =
          mlirBuilder.getFloatAttr(mlirBuilder.getF64Type(), value);
return mlirBuilder.create<ConstantOp>(loc, parameter);
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto ryOp = dyn_cast_or_null<quake::RyOp>(*op);
      if (!ryOp
          || ryOp.isAdj()
          || ryOp.getTargets().size() != 1
          || !ryOp.getControls().empty()
          || ryOp.getParameters().size() != 1) {
        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      IRRewriter rewriter(ryOp->getContext());
      rewriter.setInsertionPoint(ryOp);
      Location loc = ryOp.getLoc();
      Value rotation_value = ryOp.getParameters()[0];
      auto target_qubit = ryOp.getTargets();
      auto constant_op_rz = createFloats(M_PI, rewriter, loc);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{constant_op_rz},
                                      ValueRange{}, target_qubit);
                                      
      rewriter.create<quake::RxOp>(loc, false, ValueRange{rotation_value},
                                      ValueRange{}, target_qubit);

      auto constant_op_rz_2 = createFloats(M_PI/2, rewriter, loc);
      rewriter.create<quake::RzOp>(loc, false, ValueRange{constant_op_rz_2},
                                      ValueRange{}, target_qubit);

      rewriter.eraseOp(ryOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRyToRzRxRzPass() {
  return std::make_unique<RyToRzRxRz>();
}