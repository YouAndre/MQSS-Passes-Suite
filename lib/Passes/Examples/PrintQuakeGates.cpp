
#include "Passes/Examples.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

class PrintQuakeGates final
    : public PassWrapper<PrintQuakeGates, OperationPass<ModuleOp> > {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintQuakeGates)

  explicit PrintQuakeGates(llvm::raw_string_ostream &ostream)
    : outputStream(ostream) {
  }

  StringRef getArgument() const override {
    return "print-quake-gates-pass";
  }

  StringRef getDescription() const override {
    return "Example pass that traverses a given mlir module, print its gates "
        "and a description of the operands of each gate";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    circuit.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() == "quake") {
        outputStream << "Quantum Operation: " << op->getName().getStringRef()
            << "\n";

        // Iterate over the operands (qubits) the operation acts on
        for (Value operand : op->getOperands()) {
          if (operand.getType()
            .isa<quake::RefType>()) {
            // Check if it's a qubit reference
            outputStream << "  Acts on qubit: " << operand << "\n";
          }
        }
      }
    });
  }

private:
  llvm::raw_string_ostream &outputStream; // Store the output stream
};

} // namespace

std::unique_ptr<Pass>
mqss::opt::createPrintQuakeGatesPass(llvm::raw_string_ostream &ostream) {
  return std::make_unique<PrintQuakeGates>(ostream);
}