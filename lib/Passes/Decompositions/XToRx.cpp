#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_XTORX

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;
using mlir::arith::ConstantOp;
namespace {
class XToRx final : public BaseMQSSPass<XToRx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XToRx)

  StringRef getArgument() const override { return "XToRx"; }

  StringRef getDescription() const override {
    return "Decomposition pass of X by Rx";
  }

  auto createFloats(double value, OpBuilder &mlirBuilder, Location loc) {

      auto parameter =
          mlirBuilder.getFloatAttr(mlirBuilder.getF64Type(), value);
return mlirBuilder.create<ConstantOp>(loc, parameter);
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto xOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!xOp
        || xOp.getTargets().size() != 1
        || !xOp.getControls().empty()) {
        return;
      }
      //hop->printing().debug("Decomposing H gate into Rz-Rx-Rz sequence");
      //Qqubit target_qubit = targets[0];
      IRRewriter rewriter(xOp->getContext());
      rewriter.setInsertionPoint(xOp);
      Location loc = xOp.getLoc();
      auto target_qubit = xOp.getTargets();
      auto constant_op_rx = createFloats(M_PI, rewriter, loc);

      rewriter.create<quake::RxOp>(loc, false, ValueRange{constant_op_rx}, ValueRange{}, target_qubit);
      rewriter.eraseOp(xOp);
      this->wasApplied->store(true);                                

    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createXToRxPass() {
  return std::make_unique<XToRx>();
}
