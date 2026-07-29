#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_R1TORZ

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class R1ToRz final : public BaseMQSSPass<R1ToRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(R1ToRz)

  StringRef getArgument() const override { return "R1ToRz"; }

  StringRef getDescription() const override {
    return "Decomposition pass that replaces R1 by Rz and 1";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto r1Op = dyn_cast_or_null<quake::R1Op>(*op);
      if (!r1Op
          || r1Op.isAdj()
          || r1Op.getTargets().size() != 1
          || !r1Op.getControls().empty()
          || r1Op.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(r1Op->getContext());
      Value target = r1Op.getTargets()[0];
      Value param = r1Op.getParameters()[0];
      Location loc = r1Op.getLoc();
      rewriter.setInsertionPointAfter(r1Op);
      rewriter.create<quake::RzOp>(loc, false, param, ValueRange{}, target);
      rewriter.eraseOp(r1Op);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createR1ToRzPass() {
  return std::make_unique<R1ToRz>();
}