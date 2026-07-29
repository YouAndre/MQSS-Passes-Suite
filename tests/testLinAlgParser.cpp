

#include <iostream>
#include <string>
// llvm includes
#include "llvm/Support/raw_ostream.h"
// mlir includes
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h" // For translateModuleToLLVMIR
#include "mlir/Transforms/Passes.h"
// includes in runtime
#include "common/RuntimeMLIR.h"
// includes mqss passes
#include "Passes/CodeGen.hpp"
#include "Passes/Transforms.hpp"
// test includes
#include <fstream>
#include <gtest/gtest.h>
#include <mlir_utils.hpp>

#define CUDAQ_GEN_PREFIX_NAME "__nvqpp__mlirgen__"

using namespace mqss::support::quakeDialect;

std::tuple<ModuleOp, MLIRContext *> createEmptyMLIRModule() {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();
  // Create an empty MLIR module
  mlir::OwningOpRef m_module =
      ModuleOp::create(mlir::UnknownLoc::get(&context));
  return std::make_tuple(m_module.release(), contextPtr.release());
}

std::tuple<std::string, std::string>
getQuakeAndGolden(const std::string &inputFile, const std::string &goldenFile) {
  std::string quakeModule = readFileToString(inputFile);
  std::string goldenOutput = readFileToString(goldenFile);
  return std::make_tuple(quakeModule, goldenOutput);
}

std::string normalize(const std::string &str) {
  std::string result;
  for (char c : str) {
    if (c != '\t' && c != '\n' && c != '\\' && c != ' ') {
      result += c;
    }
  }
  return result;
}

TEST(TestLinAlgPass, TestQuakeToLinAlg) {
  // load mlir module and the golden output
  auto [quakeModule, goldenOutput] =
      getQuakeAndGolden("./quake/QuakeToLinAlgPass.qke",
                        "./golden-cases/QuakeToTikzPass.tikz.tex");
#ifdef DEBUG
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
#endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  mlir::MLIRContext &context = *contextPtr;
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::complex::ComplexDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding custom pass
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mqss::opt::createQuakeToLinAlgPass());
  // running the pass
  if (mlir::failed(pm.run(mlirModule))) {
    throw std::runtime_error("The pass failed...");
  }
#ifdef DEBUG
  std::cout << "Captured output from Pass:\n" << std::endl;
  mlirModule->dump();
#endif
  //  EXPECT_EQ(normalize(goldenOutput), normalize(moduleOutput));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}