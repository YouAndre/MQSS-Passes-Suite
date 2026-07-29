
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
#define GEN_PASS_DEF_CRZTORZCU3

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class CrzToRzCu3 final : public BaseMQSSPass<CrzToRzCu3>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrzToRzCu3)

  StringRef getArgument() const override { return "CrzToRzCu3"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Crz by a controlled U3 gate, with a "
           "compensating Rz correction on the control qubit to preserve "
           "the relative phase";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto crzOp = dyn_cast_or_null<quake::RzOp>(*op);
      if (!crzOp
          || crzOp.isAdj()
          || crzOp.getTargets().size() != 1
          || crzOp.getControls().size() != 1
          || crzOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(crzOp->getContext());
      rewriter.setInsertionPointAfter(crzOp);
      Value target = crzOp.getTargets()[0];
      Value param = crzOp.getParameters()[0];
      Value control = crzOp.getControls()[0];
      Location loc = crzOp.getLoc();

      // Rz(angle) = e^{-i*angle/2} * U3(0,0,angle): the two differ by a
      // phase that is unobservable on an uncontrolled gate, but once
      // controlled it becomes a *relative* phase between the control's |0>
      // and |1> subspaces (Crz(angle) = diag(1,1,e^{-i*angle/2},e^{i*angle/2})
      // vs. C-U3(0,0,angle) = diag(1,1,1,e^{i*angle})). Naively swapping in a
      // controlled U3 here would silently produce the wrong unitary for any
      // circuit where the control qubit is in superposition (QFT, phase
      // estimation, ...). Compensating with an uncontrolled Rz(-angle/2) on
      // the control qubit cancels exactly that discrepancy, up to an overall
      // global phase on the whole circuit, which is unobservable.
      std::vector<double> params =
          mqss::support::quakeDialect::getOperationParameters(crzOp);
      double angle = params[0];

      Value constant_0 =
          mqss::support::quakeDialect::createFloatValue(rewriter, loc, 0.0);
      Value control_phase_correction =
          mqss::support::quakeDialect::createFloatValue(rewriter, loc,
                                                          -angle / 2);

      rewriter.create<quake::RzOp>(loc, false, control_phase_correction,
                                    ValueRange{}, control);
      rewriter.create<quake::U3Op>(loc, ValueRange{constant_0, constant_0, param},
                                    control, target);
      rewriter.eraseOp(crzOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createCrzToRzCu3Pass() {
  return std::make_unique<CrzToRzCu3>();
}
