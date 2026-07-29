

// QCEC checker headers
#include "EquivalenceCheckingManager.hpp"
#include "EquivalenceCriterion.hpp"

#include <iostream>
#include <string>
// llvm includes
#include "llvm/Support/raw_ostream.h"
// mlir includes
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h" // For translateModuleToLLVMIR
#include "mlir/Transforms/Passes.h"
// includes in runtime
#include "common/RuntimeMLIR.h"
// test includes
#include "Passes/Cancellations.hpp"
#include "Passes/Decompositions.hpp"
#include "Passes/Transforms.hpp"
#include "mlir_utils.hpp"

#include <gtest/gtest.h>

#define CUDAQ_GEN_PREFIX_NAME "__nvqpp__mlirgen__"

using namespace mqss::opt;
using namespace mqss::support::quakeDialect;

std::string normalize(const std::string &str) {
  std::string result;
  for (const char c : str) {
    if (c != '\t' && c != '\n' && c != '\\' && c != ' ') {
      result += c;
    }
  }
  return result;
}

std::string lowerQuakeCodeToOpenQASM(const std::string &quantumTask) {
  auto [m_module, contextPtr] = extractMLIRContext(quantumTask);

  const std::string postCodeGenPasses = "";

  const auto translation = cudaq::getTranslation("qasm2");
  std::string codeStr;
  {
    constexpr bool enablePrintMLIREachPass = false;
    constexpr bool enablePassStatistics = false;
    llvm::raw_string_ostream outStr(codeStr);
    m_module.getContext()->disableMultithreading();
    if (constexpr bool printIR = false; mlir::failed(
            translation(m_module, outStr, postCodeGenPasses, printIR,
                        enablePrintMLIREachPass, enablePassStatistics)))
      throw std::runtime_error("Could not successfully translate to OpenQASM2");
  }
  // Regular expression to match the gate definition
  const std::regex gatePattern(R"(gate\s+\S+\(param0\)\s*\{\n\})");
  // Remove the matching part from the string
  codeStr = std::regex_replace(codeStr, gatePattern, "");
  return codeStr;
}

class EqualityTest : public testing::Test {
  void SetUp() override {
    qc1 = qc::QuantumComputation();
    qc2 = qc::QuantumComputation();

    config.execution.runSimulationChecker = false;
    config.execution.runAlternatingChecker = false;
    config.execution.runConstructionChecker = false;
    config.execution.runZXChecker = false;
  }

protected:
  // std::size_t nqubits = 1U;
  qc::QuantumComputation qc1;
  qc::QuantumComputation qc2;
  ec::Configuration config{};

  std::tuple<std::string, std::string, std::string,
             std::function<std::unique_ptr<mlir::Pass>()>>
      passInfo;
};

// Return QASM strings of the input module and the module after pass
std::tuple<std::string, std::string>
verificationTest(std::tuple<std::string, std::string,
                            std::function<std::unique_ptr<mlir::Pass>()>>
                     test) {
  auto [ignored, fileInputTest, passMlir] = test;

  // Invoke the function to create the pass
  std::unique_ptr<mlir::Pass> pass = passMlir();

  // load mlir module
  std::string quakeModule = readFileToString(fileInputTest);
  // get the QASM of the input module
  std::string qasmInput = lowerQuakeCodeToOpenQASM(quakeModule);

#ifdef DEBUG
  std::cout << "Input Quake Module " << std::endl << quakeModule << std::endl;
  std::cout << "QASM input module:" << std::endl << qasmInput << std::endl;
#endif
  auto [mlirModule, contextPtr] = extractMLIRContext(quakeModule);
  MLIRContext &context = *contextPtr;
  // creating pass manager
  mlir::PassManager pm(&context);
  // Adding the pass to the PassManager
  pm.addPass(std::move(pass));
  // pass to canonical form and remove non-used operations
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // running the pass
  if (mlir::failed(pm.run(mlirModule))) {
    throw std::runtime_error("The pass failed...");
  }
#ifdef DEBUG
  std::cout << "Circuit after pass:\n";
  mlirModule->dump();
#endif
  // Convert the module to a string
  std::string moduleOutput;
  llvm::raw_string_ostream stringStream(moduleOutput);
  mlirModule->print(stringStream);
  // dump output to qasm
  std::string qasmOutput = lowerQuakeCodeToOpenQASM(moduleOutput);
#ifdef DEBUG
  std::cout << "QASM output module " << std::endl << qasmOutput << std::endl;
#endif
  return std::make_tuple(qasmInput, qasmOutput);
}

class VerificationTestPassesMQSS
    : public ::testing::TestWithParam<
          std::tuple<std::string, std::string,
                     std::function<std::unique_ptr<mlir::Pass>()>>> {};

TEST_P(VerificationTestPassesMQSS, Run) {
  std::tuple<std::string, std::string,
             std::function<std::unique_ptr<mlir::Pass>()>>
      p = GetParam();
  std::string testName = std::get<0>(p);
  SCOPED_TRACE(testName);
  auto [qasmInput, qasmOutput] = verificationTest(p);
  // qcec objects required for verification
  qc::QuantumComputation qc1, qc2;
  ec::Configuration config{};
  auto qasmStream = std::stringstream(qasmInput);
  qc1.import(qasmStream, qc::Format::OpenQASM2);
  qasmStream = std::stringstream(qasmOutput);
  qc2.import(qasmStream, qc::Format::OpenQASM2);
  // set the configuration
  config.functionality.traceThreshold = 1e-2;
  config.execution.runConstructionChecker = true;
  config.execution.runAlternatingChecker = false;
  config.execution.runZXChecker = false;
  config.execution.runSimulationChecker = true;
  ec::EquivalenceCheckingManager ecm(qc1, qc2, config);
  ecm.run();
  std::cout << ecm.getResults() << "\n";
  EXPECT_EQ(ecm.equivalence(), ec::EquivalenceCriterion::Equivalent);
}

INSTANTIATE_TEST_SUITE_P(
    MQSSPassTests, VerificationTestPassesMQSS,
    ::testing::Values(
        std::make_tuple("TestCxToUpperHCzHPass", "./quake/CxToHCzHPass.qke",
                        []() { return mqss::opt::createCxToUpperHCzHPass(); }),
        std::make_tuple("TestCxToLowerHCzHPass", "./quake/CxToHCzHPass.qke",
                        []() { return mqss::opt::createCxToLowerHCzHPass(); }),
        std::make_tuple("TestCzToUpperHCxHPass", "./quake/CzToHCxHPass.qke",
                        []() { return mqss::opt::createCzToUpperHCxHPass(); }),
        std::make_tuple("TestCzToLowerHCxHPass", "./quake/CzToHCxHPass.qke",
                        []() { return mqss::opt::createCzToLowerHCxHPass(); }),
        std::make_tuple("TestCommuteCnotRxPass",
                        "./quake/CommuteCNotRxPass.qke",
                        []() { return mqss::opt::createCxRxToRxCxPass(); }),
        std::make_tuple("TestCommuteCnotXPass", "./quake/CommuteCNotXPass.qke",
                        []() { return mqss::opt::createCxXToXCxPass(); }),
        std::make_tuple("TestCommuteCnotZPass01",
                        "./quake/CommuteCNotZPass-01.qke",
                        []() { return mqss::opt::createCxZToZCxPass(); }),
        std::make_tuple("TestCommuteCnotZPass", "./quake/CommuteCNotZPass.qke",
                        []() { return mqss::opt::createCxZToZCxPass(); }),
        std::make_tuple("TestCommuteRxCnotPass",
                        "./quake/CommuteRxCNotPass.qke",
                        []() { return mqss::opt::createRxCxToCxRxPass(); }),
        std::make_tuple("TestCommuteXCNotPass", "./quake/CommuteXCNotPass.qke",
                        []() { return mqss::opt::createXCxToCxXPass(); }),
        std::make_tuple("TestCommuteZCnotPass", "./quake/CommuteZCNotPass.qke",
                        []() { return mqss::opt::createZCxToCxZPass(); }),
        std::make_tuple("TestCommuteZCnotPass01",
                        "./quake/CommuteZCNotPass-01.qke",
                        []() { return mqss::opt::createZCxToCxZPass(); }),
        std::make_tuple("CxCxToIdPass", "./quake/CxCxToIdPass.qke",
                        []() { return mqss::opt::createCxCxToIdPass(); }),
        std::make_tuple("ReverseCNotPass", "./quake/ReverseCNotPass.qke",
                        []() { return mqss::opt::createReverseCxPass(); }),
        std::make_tuple("HXHToZPass", "./quake/HXHToZPass.qke",
                        []() { return mqss::opt::createHXHToZPass(); }),
        std::make_tuple("XGateAndHadamardSwitchPass",
                        "./quake/XGateAndHadamardSwitchPass.qke",
                        []() { return mqss::opt::createXHToHZPass(); }),
        std::make_tuple("HZHToXPass", "./quake/HZHToXPass.qke",
                        []() { return mqss::opt::createHZHToXPass(); }),
        std::make_tuple("HadamardAndXGateSwitchPass",
                        "./quake/HadamardAndXGateSwitchPass.qke",
                        []() { return mqss::opt::createHXToZHPass(); }),
        std::make_tuple("HadamardAndZGateSwitchPass",
                        "./quake/HadamardAndZGateSwitchPass.qke",
                        []() { return mqss::opt::createHZToXHPass(); }),
        std::make_tuple("ZeroRxToIdPass", "./quake/ZeroRxToIdPass.qke",
                        []() { return mqss::opt::createZeroRxToIdPass(); }),
        std::make_tuple("ZeroRyToIdPass", "./quake/ZeroRyToIdPass.qke",
                        []() { return mqss::opt::createZeroRyToIdPass(); }),
        std::make_tuple("ZeroRzToIdPass", "./quake/ZeroRzToIdPass.qke",
                        []() { return mqss::opt::createZeroRzToIdPass(); }),
        std::make_tuple("SdgZToSPass", "./quake/SAdjToSPass.qke",
                        []() { return mqss::opt::createSdgZToSPass(); }),
        std::make_tuple("SZToSdgPass", "./quake/SToSAdjPass.qke",
                        []() { return mqss::opt::createSZToSdgPass(); })),
    [](const ::testing::TestParamInfo<VerificationTestPassesMQSS::ParamType>
           &info) {
      // Use the first element of the tuple (testName) as the custom test name
      return std::get<0>(info.param);
    });

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}