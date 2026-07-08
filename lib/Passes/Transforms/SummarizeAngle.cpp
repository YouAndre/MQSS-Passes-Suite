#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

#include <atomic>
#include <cmath>
#include <numbers>

namespace mqss::opt {
#define GEN_PASS_DEF_SUMMARIZEANGLE

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::quakeDialect;

namespace {

double normalizeAngle(double angle) {
  constexpr double twoPi = 2 * std::numbers::pi;
  return angle - std::floor(angle / twoPi) * twoPi;
}

// Merges two consecutive rotations of the same kind (OpT: RxOp, RyOp, RzOp
// or R1Op) on the same target and, if present, the same single control
// qubit into a single normalized rotation. Adjoint rotations are folded in
// as a negated angle (op^dagger(theta) == op(-theta)) before summing.
template <typename OpT>
bool summarizeRotationPair(Operation *op, std::atomic_bool &was_applied) {
  auto op2 = dyn_cast_or_null<OpT>(*op);
  if (!op2 || op2.getTargets().size() != 1 || op2.getControls().size() > 1 ||
      op2.getParameters().size() != 1) {
    return false;
  }
  auto optionalOp1 = getPreviousOperationOnTarget(op2, op2.getTargets()[0]);
  if (!optionalOp1) {
    return false;
  }
  auto op1 = dyn_cast_or_null<OpT>(*optionalOp1);
  if (!op1 || op1.getTargets().size() != 1 ||
      op1.getControls().size() != op2.getControls().size() ||
      op1.getParameters().size() != 1) {
    return false;
  }
  if (!op2.getControls().empty()) {
    auto optionalOp1OnControl =
        getPreviousOperationOnTarget(op2, op2.getControls()[0]);
    // Compare controls by qubit index rather than by SSA Value: each use of
    // a qubit typically gets its own quake.extract_ref, so two references to
    // the same physical qubit are rarely the same Value.
    auto controlIdx1 = extractIndexFromQuakeExtractRefOp(
        op1.getControls()[0].getDefiningOp());
    auto controlIdx2 = extractIndexFromQuakeExtractRefOp(
        op2.getControls()[0].getDefiningOp());
    if (optionalOp1OnControl != optionalOp1 || !controlIdx1 || !controlIdx2 ||
        *controlIdx1 != *controlIdx2) {
      return false;
    }
  }
  std::vector<double> op1Params = getOperationParameters(op1);
  std::vector<double> op2Params = getOperationParameters(op2);
  if (op1Params.size() != 1 || op2Params.size() != 1) {
    return false;
  }
  double angle1 = op1.isAdj() ? -op1Params[0] : op1Params[0];
  double angle2 = op2.isAdj() ? -op2Params[0] : op2Params[0];
  double angle = normalizeAngle(angle1 + angle2);

  IRRewriter rewriter(op2->getContext());
  rewriter.setInsertionPointAfter(op2);
  Location loc = op1.getLoc();
  Value param = createFloatValue(rewriter, loc, angle);
  rewriter.create<OpT>(loc, /*is_adj=*/false, ValueRange{param},
                        op2.getControls(), op1.getTargets());
  rewriter.eraseOp(op1);
  rewriter.eraseOp(op2);
  was_applied.store(true);
  return true;
}

class SummarizeAngle final : public BaseMQSSPass<SummarizeAngle>,
                              public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SummarizeAngle)

  StringRef getArgument() const override { return "SummarizeAngle"; }

  StringRef getDescription() const override {
    return "Collapse consecutive Rx, Ry, Rz and R1 rotations (and their "
           "single-controlled variants) on the same target/control into one "
           "angle-summed, normalized rotation.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      summarizeRotationPair<quake::RxOp>(op, *wasApplied) ||
          summarizeRotationPair<quake::RyOp>(op, *wasApplied) ||
          summarizeRotationPair<quake::RzOp>(op, *wasApplied) ||
          summarizeRotationPair<quake::R1Op>(op, *wasApplied);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSummarizeAnglePass() {
  return std::make_unique<SummarizeAngle>();
}
