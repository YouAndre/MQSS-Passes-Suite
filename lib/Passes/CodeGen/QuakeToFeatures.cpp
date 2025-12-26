#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Support/Graph.hpp"
#include "Passes/Examples.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <array>

// Include auto-generated pass registration
namespace mqss::opt {
 

#define GEN_PASS_DEF_CRXTOHCRZH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class QuakeToFeatures final : public BaseMQSSPass<QuakeToFeatures>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeToFeatures)

  StringRef getArgument() const override { return "QuakeToFeatures"; }

  StringRef getDescription() const override {
    return "QuakeToFeatures pass that extracts features from a quake kernel";
  }


struct CircuitFeatures {
  int64_t totalQubits = 0;
  int64_t totalGates = 0;
  int64_t singleQubitGates = 0;
  int64_t twoQubitGates = 0;
  int64_t parameterizedGates = 0;
  int64_t circuitDepth = 0;
  int64_t measurementCount = 0;

  enum GateIndex : uint32_t {
    H,
    X, Y, Z,
    CX, CY, CZ,
    SWAP,
    S, SDG,
    T, TDG,
    U2, U3, CU3,
    RX, CRX,
    RY, CRY,
    RZ, CRZ,
    R1,
    NUM_GATES
  };
  std::vector<uint32_t> gateCounts =
      std::vector<uint32_t>(GateIndex::NUM_GATES, 0);
};
static int getQubitIndex(mlir::Value v){
//TODO

};

static void classifyAndCount(Operation *op,
                             CircuitFeatures &features,
                             Graph &circuit_graph,
                             Graph &interaction_graph) {
  using GI = CircuitFeatures::GateIndex;

  if (!op->getDialect() ||
      op->getDialect()->getNamespace() != "quake")
    return;

  // ---- X / CX ----
  if (auto xOp = dyn_cast<quake::XOp>(op)) {
    features.totalGates++;
    int target = getQubitIndex(xOp.getTargets()[0]);

    if (xOp.getControls().empty()) {
      ++features.gateCounts[GI::X];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(xOp.getControls()[0]);
      ++features.gateCounts[GI::CX];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }

  // ---- Y / CY ----
  if (auto yOp = dyn_cast<quake::YOp>(op)) {
    features.totalGates++;
    int target = getQubitIndex(yOp.getTargets()[0]);

    if (yOp.getControls().empty()) {
      ++features.gateCounts[GI::Y];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(yOp.getControls()[0]);
      ++features.gateCounts[GI::CY];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }

  // ---- Z / CZ ----
  if (auto zOp = dyn_cast<quake::ZOp>(op)) {
    features.totalGates++;
    int target = getQubitIndex(zOp.getTargets()[0]);

    if (zOp.getControls().empty()) {
      ++features.gateCounts[GI::Z];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(zOp.getControls()[0]);
      ++features.gateCounts[GI::CZ];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }

  // ---- H ----
  if (auto hOp = dyn_cast<quake::HOp>(op)) {
    features.totalGates++;
    ++features.gateCounts[GI::H];
    features.singleQubitGates++;
    circuit_graph.create_node({getQubitIndex(hOp.getTargets()[0])});
    return;
  }

  // ---- S / SDG ----
  if (auto sOp = dyn_cast<quake::SOp>(op)) {
    features.totalGates++;
    features.singleQubitGates++;
    circuit_graph.create_node({getQubitIndex(sOp.getTargets()[0])});

    if (sOp.isAdj())
      ++features.gateCounts[GI::SDG];
    else
      ++features.gateCounts[GI::S];
    return;
  }

  // ---- T / TDG ----
  if (auto tOp = dyn_cast<quake::TOp>(op)) {
    features.totalGates++;
    features.singleQubitGates++;
    circuit_graph.create_node({getQubitIndex(tOp.getTargets()[0])});

    if (tOp.isAdj())
      ++features.gateCounts[GI::TDG];
    else
      ++features.gateCounts[GI::T];
    return;
  }

  // ---- SWAP ----
  if (auto swapOp = dyn_cast<quake::SwapOp>(op)) {
    features.totalGates++;
    features.twoQubitGates++;
    ++features.gateCounts[GI::SWAP];

    int q0 = getQubitIndex(swapOp.getTargets()[0]);
    int q1 = getQubitIndex(swapOp.getTargets()[1]);

    circuit_graph.create_node({q0, q1});
    interaction_graph.create_interaction_node(q0, q1);
    interaction_graph.create_interaction_node(q1, q0);
    return;
  }

  // ---- RX / CRX ----
  if (auto rxOp = dyn_cast<quake::RxOp>(op)) {
    features.totalGates++;
    features.parameterizedGates++;
    int target = getQubitIndex(rxOp.getTargets()[0]);

    if (rxOp.getControls().empty()) {
      ++features.gateCounts[GI::RX];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(rxOp.getControls()[0]);
      ++features.gateCounts[GI::CRX];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }

  // ---- RY / CRY ----
  if (auto ryOp = dyn_cast<quake::RyOp>(op)) {
    features.totalGates++;
    features.parameterizedGates++;
    int target = getQubitIndex(ryOp.getTargets()[0]);

    if (ryOp.getControls().empty()) {
      ++features.gateCounts[GI::RY];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(ryOp.getControls()[0]);
      ++features.gateCounts[GI::CRY];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }

  // ---- RZ / CRZ ----
  if (auto rzOp = dyn_cast<quake::RzOp>(op)) {
    features.totalGates++;
    features.parameterizedGates++;
    int target = getQubitIndex(rzOp.getTargets()[0]);

    if (rzOp.getControls().empty()) {
      ++features.gateCounts[GI::RZ];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(rzOp.getControls()[0]);
      ++features.gateCounts[GI::CRZ];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }

  // ---- R1 ----
  if (auto r1Op = dyn_cast<quake::R1Op>(op)) {
    features.totalGates++;
    features.parameterizedGates++;
    features.singleQubitGates++;
    ++features.gateCounts[GI::R1];
    circuit_graph.create_node({getQubitIndex(r1Op.getTargets()[0])});
    return;
  }

  // ---- U2 ----
  if (auto u2Op = dyn_cast<quake::U2Op>(op)) {
    features.totalGates++;
    features.parameterizedGates++;
    features.singleQubitGates++;
    ++features.gateCounts[GI::U2];
    circuit_graph.create_node({getQubitIndex(u2Op.getTargets()[0])});
    return;
  }

  // ---- U3 / CU3 ----
  if (auto u3Op = dyn_cast<quake::U3Op>(op)) {
    features.totalGates++;
    features.parameterizedGates++;
    int target = getQubitIndex(u3Op.getTargets()[0]);

    if (u3Op.getControls().empty()) {
      ++features.gateCounts[GI::U3];
      features.singleQubitGates++;
      circuit_graph.create_node({target});
    } else {
      int control = getQubitIndex(u3Op.getControls()[0]);
      ++features.gateCounts[GI::CU3];
      features.twoQubitGates++;
      circuit_graph.create_node({control, target});
      interaction_graph.create_interaction_node(control, target);
    }
    return;
  }
}




  void runOnOperation() override {
    FuncOp func = getOperation();
    if (!func->hasAttr("cudaq-entrypoint")) return;  // Skip non-kernels
    

    CircuitFeatures features;
    std::unique_ptr<Graph> circuit_graph;
    std::unique_ptr<Graph> interaction_graph;

    // Walk all operations in the function
    func.walk([&](Operation* op) {
      // Qubit allocation (total qubits)
      if (auto allocaOp = dyn_cast<quake::AllocaOp>(op)) {
        if (auto veq = dyn_cast<quake::VeqType>(allocaOp.getType())) {
          if (auto sz = veq.getSize()) {
            features.totalQubits += sz;
            //WARNING ASSUME only 1 allocation per kernel for now
            if (!circuit_graph) {
            circuit_graph = std::make_unique<Graph>(features.totalQubits);
            interaction_graph = std::make_unique<Graph>(features.totalQubits);
          }
          }
        } 
      }
      else {
        classifyAndCount(op, features,*circuit_graph,
                       *interaction_graph);
      }
      auto [depth,multiqubit] = circuit_graph->get_max_depths();
      auto [pc,dpc] = interaction_graph->compute_program_communication();
      features.circuitDepth = depth;
      features.programCommunication = pc;
      features.directedProgramCommunication = dpc;



      

    });}




 // namespace
  }
std::unique_ptr<Pass> mqss::opt::createQuakeToFeaturesPass() {
  return std::make_unique<QuakeToFeatures>();
}