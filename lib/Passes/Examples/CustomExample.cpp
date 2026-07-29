
#include "Passes/Examples.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Here is an example MLIR Pass that one can write externally and
// use via the cudaq-opt tool, with the --load-cudaq-plugin flag.
// The pass here is simple, replace Hadamard operations with S operations.

using namespace mlir;

namespace {

struct ReplaceH final : OpRewritePattern<quake::HOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::HOp hOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<quake::SOp>(
        hOp, hOp.isAdj(), hOp.getParameters(), hOp.getControls(),
        hOp.getTargets());
    return success();
  }
};

class CustomExamplePassPlugin final
    : public PassWrapper<CustomExamplePassPlugin, OperationPass<ModuleOp> > {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomExamplePassPlugin)

  StringRef getArgument() const override { return "cudaq-custom-pass"; }

  StringRef getDescription() const override {
    return "Apply dummy example pass that replaces each H gate by a S gate";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    auto ctx = circuit.getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceH>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::HOp>();
    if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
      circuit.emitOpError("simple pass failed");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCustomExamplePass() {
  return std::make_unique<CustomExamplePassPlugin>();
}