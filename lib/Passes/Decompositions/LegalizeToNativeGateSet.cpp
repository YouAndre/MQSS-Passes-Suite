#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir_utils.hpp"
#include "llvm/ADT/StringSet.h"
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_LEGALIZETONATIVEGATESET

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Rewrite rules.
//
// Each function below reproduces, verbatim, the rewrite performed by the
// corresponding single-purpose pass in this directory (e.g. rewriteXToRx()
// mirrors XToRx.cpp). They are kept separate from those passes rather than
// calling into them because each one only walks/matches its own op; here we
// already know which rule matched a given op and just need to apply it.
//===----------------------------------------------------------------------===//

void rewriteHToRzXRz(IRRewriter &rewriter, quake::HOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value c1 =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{c1}, ValueRange{},
                                target);
  rewriter.create<quake::XOp>(loc, false, ValueRange{}, ValueRange{}, target);
  Value c2 =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{c2}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteHToU3(IRRewriter &rewriter, quake::HOp op) {
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  Value halfPi =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  Value zero =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, 0.0);
  Value pi = mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI);
  rewriter.create<quake::U3Op>(loc, ValueRange{halfPi, zero, pi}, ValueRange{},
                                ValueRange{target});
  rewriter.eraseOp(op);
}

void rewriteXToHZH(IRRewriter &rewriter, quake::XOp op) {
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, false, target);
  rewriter.create<quake::ZOp>(loc, false, target);
  rewriter.create<quake::HOp>(loc, false, target);
  rewriter.eraseOp(op);
}

void rewriteXToRx(IRRewriter &rewriter, quake::XOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI);
  rewriter.create<quake::RxOp>(loc, false, ValueRange{angle}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteYToRy(IRRewriter &rewriter, quake::YOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, -M_PI);
  rewriter.create<quake::RyOp>(loc, false, ValueRange{angle}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteZToHXH(IRRewriter &rewriter, quake::ZOp op) {
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, false, target);
  rewriter.create<quake::XOp>(loc, false, target);
  rewriter.create<quake::HOp>(loc, false, target);
  rewriter.eraseOp(op);
}

void rewriteZToRz(IRRewriter &rewriter, quake::ZOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{angle}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteSToRz(IRRewriter &rewriter, quake::SOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  rewriter.create<quake::RzOp>(loc, false, angle, ValueRange{}, target);
  rewriter.eraseOp(op);
}

void rewriteSToSdgSdgSdg(IRRewriter &rewriter, quake::SOp op) {
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.eraseOp(op);
}

void rewriteSToTT(IRRewriter &rewriter, quake::SOp op) {
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::TOp>(loc, false, target);
  rewriter.create<quake::TOp>(loc, false, target);
  rewriter.eraseOp(op);
}

void rewriteSdgToRz(IRRewriter &rewriter, quake::SOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, -M_PI_2);
  rewriter.create<quake::RzOp>(loc, false, angle, ValueRange{}, target);
  rewriter.eraseOp(op);
}

void rewriteSdgToSSS(IRRewriter &rewriter, quake::SOp op) {
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::SOp>(loc, false, target);
  rewriter.create<quake::SOp>(loc, false, target);
  rewriter.create<quake::SOp>(loc, false, target);
  rewriter.eraseOp(op);
}

void rewriteTToRz(IRRewriter &rewriter, quake::TOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_4);
  rewriter.create<quake::RzOp>(loc, false, angle, ValueRange{}, target);
  rewriter.eraseOp(op);
}

void rewriteTdgToRz(IRRewriter &rewriter, quake::TOp op) {
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value angle =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, -M_PI_4);
  rewriter.create<quake::RzOp>(loc, false, angle, ValueRange{}, target);
  rewriter.eraseOp(op);
}

void rewriteR1ToRz(IRRewriter &rewriter, quake::R1Op op) {
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::RzOp>(loc, false, param, ValueRange{}, target);
  rewriter.eraseOp(op);
}

void rewriteRxToHRzH(IRRewriter &rewriter, quake::RxOp op) {
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::RzOp>(loc, false, param, ValueRange{}, target);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.eraseOp(op);
}

void rewriteRyToRzRxRz(IRRewriter &rewriter, quake::RyOp op) {
  Location loc = op.getLoc();
  Value rotation = op.getParameters()[0];
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value c1 =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{c1}, ValueRange{},
                                target);
  rewriter.create<quake::RxOp>(loc, false, ValueRange{rotation}, ValueRange{},
                                target);
  Value c2 =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{c2}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteRzToHRxH(IRRewriter &rewriter, quake::RzOp op) {
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::RxOp>(loc, false, param, ValueRange{}, target);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.eraseOp(op);
}

void rewriteRzToU3(IRRewriter &rewriter, quake::RzOp op) {
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  Value zero =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, 0.0);
  rewriter.create<quake::U3Op>(loc, ValueRange{zero, zero, param},
                                ValueRange{}, ValueRange{target});
  rewriter.eraseOp(op);
}

void rewriteU2ToRzRyRz(IRRewriter &rewriter, quake::U2Op op) {
  auto params = op.getParameters();
  Value phi = params[0];
  Value lambda = params[1];
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value halfPi =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{phi}, ValueRange{},
                                target);
  rewriter.create<quake::RyOp>(loc, false, ValueRange{halfPi}, ValueRange{},
                                target);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{lambda}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteU2ToU3(IRRewriter &rewriter, quake::U2Op op) {
  auto params = op.getParameters();
  Value phi = params[0];
  Value lambda = params[1];
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  Value halfPi =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  rewriter.create<quake::U3Op>(loc, ValueRange{halfPi, phi, lambda},
                                ValueRange{}, target);
  rewriter.eraseOp(op);
}

void rewriteU3ToRzRyRz(IRRewriter &rewriter, quake::U3Op op) {
  auto params = op.getParameters();
  Value theta = params[0];
  Value phi = params[1];
  Value lambda = params[2];
  Location loc = op.getLoc();
  ValueRange target = op.getTargets();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{lambda}, ValueRange{},
                                target);
  rewriter.create<quake::RyOp>(loc, false, ValueRange{theta}, ValueRange{},
                                target);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{phi}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

void rewriteSwapToUpperCxCxCx(IRRewriter &rewriter, quake::SwapOp op) {
  Value q0 = op.getTargets()[0];
  Value q1 = op.getTargets()[1];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::XOp>(loc, q0, q1);
  rewriter.create<quake::XOp>(loc, q1, q0);
  rewriter.create<quake::XOp>(loc, q0, q1);
  rewriter.eraseOp(op);
}

void rewriteCxToUpperHCzH(IRRewriter &rewriter, quake::XOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::ZOp>(loc, control, target);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.eraseOp(op);
}

void rewriteCzToUpperHCxH(IRRewriter &rewriter, quake::ZOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::XOp>(loc, control, target);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.eraseOp(op);
}

void rewriteCyToSCxSdg(IRRewriter &rewriter, quake::YOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::SOp>(loc, target);
  rewriter.create<quake::XOp>(loc, control, target);
  rewriter.create<quake::SOp>(loc, true, target);
  rewriter.eraseOp(op);
}

void rewriteCrxToHCrzH(IRRewriter &rewriter, quake::RxOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::RzOp>(loc, false, param, control, target);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.eraseOp(op);
}

void rewriteCrzToHCrxH(IRRewriter &rewriter, quake::RzOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.create<quake::RxOp>(loc, false, param, control, target);
  rewriter.create<quake::HOp>(loc, target);
  rewriter.eraseOp(op);
}

void rewriteCrzToRzCu3(IRRewriter &rewriter, quake::RzOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  std::vector<double> params =
      mqss::support::quakeDialect::getOperationParameters(op);
  double angle = params[0];

  Value zero =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, 0.0);
  Value controlPhaseCorrection =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc,
                                                      -angle / 2);
  rewriter.create<quake::RzOp>(loc, false, controlPhaseCorrection,
                                ValueRange{}, control);
  rewriter.create<quake::U3Op>(loc, ValueRange{zero, zero, param}, control,
                                target);
  rewriter.eraseOp(op);
}

void rewriteCrzToRzCxRzCx(IRRewriter &rewriter, quake::RzOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  std::vector<double> params =
      mqss::support::quakeDialect::getOperationParameters(op);
  double angle = params[0];

  Value half =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, angle / 2);
  Value negHalf = mqss::support::quakeDialect::createFloatValue(
      rewriter, loc, angle / -2);
  rewriter.create<quake::RzOp>(loc, false, half, ValueRange{},
                                ValueRange{target});
  rewriter.create<quake::XOp>(loc, ValueRange{control}, ValueRange{target});
  rewriter.create<quake::RzOp>(loc, false, negHalf, ValueRange{},
                                ValueRange{target});
  rewriter.create<quake::XOp>(loc, ValueRange{control}, ValueRange{target});
  rewriter.eraseOp(op);
}

void rewriteCryToRzCrxRz(IRRewriter &rewriter, quake::RyOp op) {
  Value control = op.getControls()[0];
  Value target = op.getTargets()[0];
  Value param = op.getParameters()[0];
  Location loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  Value negHalfPi =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, -M_PI_2);
  Value halfPi =
      mqss::support::quakeDialect::createFloatValue(rewriter, loc, M_PI_2);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{negHalfPi},
                                ValueRange{}, target);
  rewriter.create<quake::RxOp>(loc, false, param, control, target);
  rewriter.create<quake::RzOp>(loc, false, ValueRange{halfPi}, ValueRange{},
                                target);
  rewriter.eraseOp(op);
}

//===----------------------------------------------------------------------===//
// The rule table: for every recognized "native gate mnemonic", the list of
// known ways to build it out of other gates. Multiple entries mean multiple
// choices exist in lib/Passes/Decompositions/ for that gate.
//===----------------------------------------------------------------------===//

struct DecompositionRule {
  const char *name;
  std::vector<std::string> produces;
  std::function<void(IRRewriter &, Operation *)> apply;
};

template <typename QuakeOpT>
DecompositionRule makeRule(const char *name, std::vector<std::string> produces,
                            void (*rewrite)(IRRewriter &, QuakeOpT)) {
  return DecompositionRule{
      name, std::move(produces), [rewrite](IRRewriter &rewriter, Operation *op) {
        rewrite(rewriter, cast<QuakeOpT>(op));
      }};
}

const std::unordered_map<std::string, std::vector<DecompositionRule>> &
getDecompositionTable() {
  static const std::unordered_map<std::string, std::vector<DecompositionRule>>
      table = [] {
        std::unordered_map<std::string, std::vector<DecompositionRule>> t;
        t["h"] = {makeRule<quake::HOp>("HToRzXRz", {"rz", "x"},
                                        rewriteHToRzXRz),
                   makeRule<quake::HOp>("HToU3", {"u3"}, rewriteHToU3)};
        t["x"] = {makeRule<quake::XOp>("XToHZH", {"h", "z"}, rewriteXToHZH),
                   makeRule<quake::XOp>("XToRx", {"rx"}, rewriteXToRx)};
        t["y"] = {makeRule<quake::YOp>("YToRy", {"ry"}, rewriteYToRy)};
        t["z"] = {makeRule<quake::ZOp>("ZToHXH", {"h", "x"}, rewriteZToHXH),
                   makeRule<quake::ZOp>("ZToRz", {"rz"}, rewriteZToRz)};
        t["s"] = {makeRule<quake::SOp>("SToRz", {"rz"}, rewriteSToRz),
                   makeRule<quake::SOp>("SToSdgSdgSdg", {"sdg"},
                                        rewriteSToSdgSdgSdg),
                   makeRule<quake::SOp>("SToTT", {"t"}, rewriteSToTT)};
        t["sdg"] = {makeRule<quake::SOp>("SdgToRz", {"rz"}, rewriteSdgToRz),
                     makeRule<quake::SOp>("SdgToSSS", {"s"},
                                          rewriteSdgToSSS)};
        t["t"] = {makeRule<quake::TOp>("TToRz", {"rz"}, rewriteTToRz)};
        t["tdg"] = {makeRule<quake::TOp>("TdgToRz", {"rz"}, rewriteTdgToRz)};
        t["r1"] = {makeRule<quake::R1Op>("R1ToRz", {"rz"}, rewriteR1ToRz)};
        t["rx"] = {
            makeRule<quake::RxOp>("RxToHRzH", {"h", "rz"}, rewriteRxToHRzH)};
        t["ry"] = {makeRule<quake::RyOp>("RyToRzRxRz", {"rz", "rx"},
                                         rewriteRyToRzRxRz)};
        t["rz"] = {
            makeRule<quake::RzOp>("RzToHRxH", {"h", "rx"}, rewriteRzToHRxH),
            makeRule<quake::RzOp>("RzToU3", {"u3"}, rewriteRzToU3)};
        t["u2"] = {makeRule<quake::U2Op>("U2ToRzRyRz", {"rz", "ry"},
                                         rewriteU2ToRzRyRz),
                    makeRule<quake::U2Op>("U2ToU3", {"u3"}, rewriteU2ToU3)};
        t["u3"] = {makeRule<quake::U3Op>("U3ToRzRyRz", {"rz", "ry"},
                                         rewriteU3ToRzRyRz)};
        t["swap"] = {makeRule<quake::SwapOp>(
            "SwapToUpperCxCxCx", {"cx"}, rewriteSwapToUpperCxCxCx)};
        t["cx"] = {makeRule<quake::XOp>("CxToUpperHCzH", {"h", "cz"},
                                        rewriteCxToUpperHCzH)};
        t["cz"] = {makeRule<quake::ZOp>("CzToUpperHCxH", {"h", "cx"},
                                        rewriteCzToUpperHCxH)};
        t["cy"] = {makeRule<quake::YOp>("CyToSCxSdg", {"s", "cx", "sdg"},
                                        rewriteCyToSCxSdg)};
        t["crx"] = {makeRule<quake::RxOp>("CrxToHCrzH", {"h", "crz"},
                                          rewriteCrxToHCrzH)};
        t["crz"] = {makeRule<quake::RzOp>("CrzToHCrxH", {"h", "crx"},
                                          rewriteCrzToHCrxH),
                     makeRule<quake::RzOp>("CrzToRzCu3", {"rz", "u3"},
                                          rewriteCrzToRzCu3),
                     makeRule<quake::RzOp>("CrzToRzCxRzCx", {"rz", "cx"},
                                          rewriteCrzToRzCxRzCx)};
        t["cry"] = {makeRule<quake::RyOp>("CryToRzCrxRz", {"rz", "crx"},
                                          rewriteCryToRzCrxRz)};
        return t;
      }();
  return table;
}

//===----------------------------------------------------------------------===//
// Classification: maps an Operation* to the native-gate-set mnemonic it
// represents, or std::nullopt if this pass doesn't recognize its shape
// (wrong number of controls/targets/parameters, or an adjoint rotation --
// none of the rules above handle adjoint Rx/Ry/Rz/R1, matching the scope of
// the individual passes they were copied from).
//===----------------------------------------------------------------------===//

std::optional<std::string> classifyOp(Operation *op) {
  if (auto g = dyn_cast<quake::HOp>(op)) {
    if (g.getControls().empty() && g.getTargets().size() == 1)
      return std::string("h");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::XOp>(op)) {
    if (g.getTargets().size() != 1)
      return std::nullopt;
    if (g.getControls().empty())
      return std::string("x");
    if (g.getControls().size() == 1)
      return std::string("cx");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::YOp>(op)) {
    if (g.isAdj() || g.getTargets().size() != 1)
      return std::nullopt;
    if (g.getControls().empty())
      return std::string("y");
    if (g.getControls().size() == 1)
      return std::string("cy");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::ZOp>(op)) {
    if (g.getTargets().size() != 1)
      return std::nullopt;
    if (g.getControls().empty())
      return std::string("z");
    if (g.getControls().size() == 1)
      return std::string("cz");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::SOp>(op)) {
    if (!g.getControls().empty() || g.getTargets().size() != 1)
      return std::nullopt;
    return std::string(g.isAdj() ? "sdg" : "s");
  }
  if (auto g = dyn_cast<quake::TOp>(op)) {
    if (!g.getControls().empty() || g.getTargets().size() != 1)
      return std::nullopt;
    return std::string(g.isAdj() ? "tdg" : "t");
  }
  if (auto g = dyn_cast<quake::R1Op>(op)) {
    if (g.isAdj() || !g.getControls().empty() || g.getTargets().size() != 1 ||
        g.getParameters().size() != 1)
      return std::nullopt;
    return std::string("r1");
  }
  if (auto g = dyn_cast<quake::RxOp>(op)) {
    if (g.isAdj() || g.getTargets().size() != 1 ||
        g.getParameters().size() != 1)
      return std::nullopt;
    if (g.getControls().empty())
      return std::string("rx");
    if (g.getControls().size() == 1)
      return std::string("crx");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::RyOp>(op)) {
    if (g.isAdj() || g.getTargets().size() != 1 ||
        g.getParameters().size() != 1)
      return std::nullopt;
    if (g.getControls().empty())
      return std::string("ry");
    if (g.getControls().size() == 1)
      return std::string("cry");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::RzOp>(op)) {
    if (g.isAdj() || g.getTargets().size() != 1 ||
        g.getParameters().size() != 1)
      return std::nullopt;
    if (g.getControls().empty())
      return std::string("rz");
    if (g.getControls().size() == 1)
      return std::string("crz");
    return std::nullopt;
  }
  if (auto g = dyn_cast<quake::U2Op>(op)) {
    if (!g.getControls().empty() || g.getTargets().size() != 1 ||
        g.getParameters().size() != 2)
      return std::nullopt;
    return std::string("u2");
  }
  if (auto g = dyn_cast<quake::U3Op>(op)) {
    if (!g.getControls().empty() || g.getTargets().size() != 1 ||
        g.getParameters().size() != 3)
      return std::nullopt;
    return std::string("u3");
  }
  if (auto g = dyn_cast<quake::SwapOp>(op)) {
    if (!g.getControls().empty() || g.getTargets().size() != 2)
      return std::nullopt;
    return std::string("swap");
  }
  return std::nullopt;
}

// For a given native set, figures out -- for every mnemonic that isn't
// itself native -- one rule that is *transitively* guaranteed to bottom out
// in native gates (not just a rule whose immediate output happens to be
// native already). This is a fixed-point over the whole table: a mnemonic is
// "resolvable" if native, or if some rule for it produces only mnemonics
// that are themselves resolvable.
//
// This is still not a cost model (it doesn't compare resolvable rules by
// depth/gate count, it just takes the first resolvable one found), but
// skipping it and only doing a one-hop lookahead is actively wrong: e.g. for
// native = {h, rx, cx}, "crz" has no rule whose *immediate* output is fully
// native, so a one-hop check always falls back to CrzToHCrxH -- which only
// ever produces "crx", whose only rule (CrxToHCrzH) produces "crz" right
// back. That pair bounces forever, and because each round wraps another H on
// each side without ever cancelling the previous ones, the circuit grows
// without bound instead of converging (confirmed by direct testing). The
// fixed point below instead finds that CrzToRzCxRzCx -> {rz, cx} works,
// because rz is itself resolvable via RzToHRxH -> {h, rx}.
std::unordered_map<std::string, const DecompositionRule *>
computeWitnesses(
    const std::unordered_map<std::string, std::vector<DecompositionRule>>
        &table,
    const llvm::StringSet<> &native) {
  std::unordered_map<std::string, const DecompositionRule *> witness;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &entry : table) {
      const std::string &mnemonic = entry.first;
      if (native.contains(mnemonic) || witness.count(mnemonic))
        continue;
      for (const auto &rule : entry.second) {
        bool allResolvable = true;
        for (const auto &produced : rule.produces) {
          if (native.contains(produced) || witness.count(produced))
            continue;
          allResolvable = false;
          break;
        }
        if (allResolvable) {
          witness[mnemonic] = &rule;
          changed = true;
          break;
        }
      }
    }
  }
  return witness;
}

class LegalizeToNativeGateSet final
    : public BaseMQSSPass<LegalizeToNativeGateSet>,
      public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeToNativeGateSet)

  LegalizeToNativeGateSet() = default;

  explicit LegalizeToNativeGateSet(ArrayRef<std::string> gates)
      : LegalizeToNativeGateSet() {
    nativeGateSet = gates;
  }

  LegalizeToNativeGateSet(const LegalizeToNativeGateSet &other)
      : BaseMQSSPass(other), AppliedCheckPass(other) {
    nativeGateSet = other.nativeGateSet;
  }

  StringRef getArgument() const override {
    return "LegalizeToNativeGateSet";
  }

  StringRef getDescription() const override {
    return "Quick-and-dirty legalization pass: repeatedly rewrites any gate "
           "that isn't a member of the given native gate set, using the "
           "same rules as the individual Decomposition passes, until every "
           "gate is native or no further progress can be made";
  }

  ListOption<std::string> nativeGateSet{
      *this, "native-gate-set",
      llvm::cl::desc(
          "Mnemonics of the gates considered native (e.g. h,cx,rz); see "
          "classifyOp() in LegalizeToNativeGateSet.cpp for the vocabulary")};

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);

    llvm::StringSet<> native;
    for (const auto &gate : nativeGateSet)
      native.insert(gate);

    const auto &table = getDecompositionTable();
    // Computed once, up front: for every non-native mnemonic that can
    // transitively bottom out in native gates, the rule that gets it there.
    // Mnemonics absent from this map (but present in the table) are known
    // gates this pass genuinely cannot legalize into the requested set --
    // warned about once below, and left untouched, rather than repeatedly
    // rewritten with a rule that can never converge.
    const auto witnesses = computeWitnesses(table, native);
    llvm::StringSet<> warnedUnresolvable;

    // Applying a witness rule can itself introduce ops whose own witness
    // rule hasn't fired yet (e.g. crz -> {rz, cx}, then rz -> {h, rx}), so
    // this still needs multiple rounds; the loop is now just mechanically
    // unwinding an acyclic chain, so it always reaches a fixed point in at
    // most table.size() rounds. The cap is only a defensive backstop.
    constexpr int maxRounds = 64;
    for (int round = 0; round < maxRounds; ++round) {
      bool changed = false;
      kernel.walk([&](Operation *op) {
        std::optional<std::string> mnemonic = classifyOp(op);
        if (!mnemonic || native.contains(*mnemonic))
          return;
        auto wit = witnesses.find(*mnemonic);
        if (wit == witnesses.end()) {
          if (table.count(*mnemonic) && warnedUnresolvable.insert(*mnemonic).second)
            kernel.emitWarning()
                << "LegalizeToNativeGateSet: '" << *mnemonic
                << "' cannot be legalized into the requested native gate "
                   "set with the known decomposition rules; leaving it "
                   "as-is";
          return;
        }
        IRRewriter rewriter(op->getContext());
        wit->second->apply(rewriter, op);
        changed = true;
        this->wasApplied->store(true);
      });
      if (!changed)
        return;
      if (round == maxRounds - 1)
        kernel.emitWarning()
            << "LegalizeToNativeGateSet: gave up after " << maxRounds
            << " rounds without reaching a fixed point";
    }
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createLegalizeToNativeGateSetPass(
    ArrayRef<std::string> native_gate_set) {
  return std::make_unique<LegalizeToNativeGateSet>(native_gate_set);
}
