#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/mlir_utils.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <atomic>
#include <cmath>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_GENERALCANCELLATIONS

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

// Finds the operation immediately preceding `op2` on its target qubit, when
// that operation is also an OpT on the same target and, if op2 has one, the
// same single control qubit. Returns a null OpT when there is no such match.
// Used as the common basis for both self-inverse and adjoint-pair
// cancellation, which differ only in what they additionally require of the
// matched pair (see cancelSelfInversePair and cancelAdjointPair below).
template <typename OpT>
OpT findMatchingPreviousOp(OpT op2) {
  if (op2.getTargets().size() != 1 || op2.getControls().size() > 1) {
    return OpT();
  }
  auto optionalOp1 = getPreviousOperationOnTarget(op2, op2.getTargets()[0]);
  if (!optionalOp1) {
    return OpT();
  }
  auto op1 = dyn_cast_or_null<OpT>(*optionalOp1);
  if (!op1 || op1.getTargets().size() != 1 ||
      op1.getControls().size() != op2.getControls().size()) {
    return OpT();
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
      return OpT();
    }
  }
  return op1;
}

// Covers XXToId, YYToId, ZZToId, HHToId (0 controls) and CxCxToId, CyCyToId,
// CzCzToId (1 control): two consecutive identical self-inverse gates on the
// same target and, if present, the same single control cancel out.
template <typename OpT>
bool cancelSelfInversePair(Operation *op, std::atomic_bool &was_applied) {
  auto op2 = dyn_cast_or_null<OpT>(*op);
  if (!op2) {
    return false;
  }
  auto op1 = findMatchingPreviousOp<OpT>(op2);
  if (!op1) {
    return false;
  }
  IRRewriter rewriter(op2->getContext());
  rewriter.eraseOp(op2);
  rewriter.eraseOp(op1);
  was_applied.store(true);
  return true;
}

// Covers SSdgToId, SdgSToId, TTdgToId, TdgTToId and their controlled variants
// (CS/CSdg, CT/CTdg): two consecutive S/Sdg (or T/Tdg) gates on the same
// target and, if present, the same single control cancel out when one is the
// adjoint of the other, regardless of order.
template <typename OpT>
bool cancelAdjointPair(Operation *op, std::atomic_bool &was_applied) {
  auto op2 = dyn_cast_or_null<OpT>(*op);
  if (!op2) {
    return false;
  }
  auto op1 = findMatchingPreviousOp<OpT>(op2);
  if (!op1 || op1.isAdj() == op2.isAdj()) {
    return false;
  }
  IRRewriter rewriter(op2->getContext());
  rewriter.eraseOp(op2);
  rewriter.eraseOp(op1);
  was_applied.store(true);
  return true;
}

// Covers ZeroRxToId, ZeroRyToId, ZeroRzToId, plus R1 and their controlled
// variants (CRx/CRy/CRz/CR1 with a zero angle): a rotation by a multiple of
// 2*pi is the identity regardless of the state of any control qubits, so it
// can always be removed outright rather than needing a matching pair.
template <typename OpT>
bool cancelZeroRotation(Operation *op, std::atomic_bool &was_applied) {
  auto rotOp = dyn_cast_or_null<OpT>(*op);
  if (!rotOp || rotOp.getTargets().size() != 1 ||
      rotOp.getParameters().size() != 1) {
    return false;
  }
  std::vector<double> params = getOperationParameters(rotOp);
  if (params.size() != 1 || !isMultipleOfTwoPi(params[0])) {
    return false;
  }
  IRRewriter rewriter(rotOp->getContext());
  rewriter.eraseOp(rotOp);
  was_applied.store(true);
  return true;
}

class GeneralCancellations final
    : public BaseMQSSPass<GeneralCancellations>,
      public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeneralCancellations)

  StringRef getArgument() const override { return "GeneralCancellations"; }

  StringRef getDescription() const override {
    return "Remove all cancellable gate pairs (X, Y, Z, H, Cx, Cy, Cz, "
           "S/Sdg, T/Tdg) and zero-angle rotations (Rx, Ry, Rz, R1) in a "
           "single pass.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      cancelSelfInversePair<quake::XOp>(op, *wasApplied) ||
          cancelSelfInversePair<quake::YOp>(op, *wasApplied) ||
          cancelSelfInversePair<quake::ZOp>(op, *wasApplied) ||
          cancelSelfInversePair<quake::HOp>(op, *wasApplied) ||
          cancelAdjointPair<quake::SOp>(op, *wasApplied) ||
          cancelAdjointPair<quake::TOp>(op, *wasApplied) ||
          cancelZeroRotation<quake::RxOp>(op, *wasApplied) ||
          cancelZeroRotation<quake::RyOp>(op, *wasApplied) ||
          cancelZeroRotation<quake::RzOp>(op, *wasApplied) ||
          cancelZeroRotation<quake::R1Op>(op, *wasApplied);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createGeneralCancellationsPass() {
  return std::make_unique<GeneralCancellations>();
}
