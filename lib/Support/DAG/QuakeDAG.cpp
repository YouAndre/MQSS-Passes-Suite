

#include "Support/DAG/Quake-DAG.hpp"
#include "Support/mlir_utils.hpp"

#include <ranges>

using namespace mqss::support::quakeDialect;

void QuakeDAG::parse_mlir(FuncOp kernel) {
  const int numQubits = getNumberOfQubits(kernel);
  std::map<size_t, Vertex>
      qubitsHistory; // this map stores the last inserted vertex in the graph on
  // each qubit, the key is the index of the qubit
  for (int i = 0; i < numQubits; i++) {
    const Vertex qubit = get_or_add_node("q_" + std::to_string(i));
    dag[qubit].isQubit = true;
    qubitsHistory[i] = qubit;
  }
  int idx = 0;
  kernel.walk([&](Operation *op) {
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    if (!gate)
      return;
    // then, the operation is a quake gate
    StringRef opName = op->getName().getStringRef();
    std::regex pattern("^quake\\.");
    std::string result = std::regex_replace(opName.str(), pattern, "");
    Vertex operation = get_or_add_node(result + "_" + std::to_string(idx++));
    std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
    std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
    std::vector<double> params = getParametersValues(gate.getParameters());
    // load data into the vertex
    dag[operation].operation = op;
    dag[operation].targets = targets;
    dag[operation].controls = controls;
    dag[operation].arguments = params;
    dag[operation].isAdj = gate.isAdj();
    // insert the edges
    for (int i = 0; i < targets.size(); i++) {
      if (qubitsHistory.contains(targets[i])) {
        add_edge(qubitsHistory[targets[i]], operation, dag);
        qubitsHistory[targets[i]] = operation;
      } else
        assert("This should not happen!");
    }
    for (int i = 0; i < controls.size(); i++) {
      if (qubitsHistory.contains(controls[i])) {
        add_edge(qubitsHistory[controls[i]], operation, dag);
        qubitsHistory[controls[i]] = operation;
      } else
        assert("This should not happen!");
    }
  });
  // TODO read the measurements from Quake file
  // insert measurements to al qubits at the end
  Vertex mes = get_or_add_node("measurement");
  dag[mes].isMeasurement = true;
  for (const auto &value : qubitsHistory | std::views::values) {
    add_edge(value, mes, dag);
  }
}

// Print the DAG to console
void QuakeDAG::print() const {
  boost::graph_traits<DAG>::vertex_iterator vi, vi_end;
  for (std::tie(vi, vi_end) = boost::vertices(dag); vi != vi_end; ++vi) {
    std::cout << dag[*vi].name << " -> ";
    for (auto out : boost::make_iterator_range(adjacent_vertices(*vi, dag))) {
      std::cout << dag[out].name << ", ";
    }
    std::cout << std::endl;
  }
}

// Dump to .dot format
void QuakeDAG::dump_dot(const std::string &filename) const {
  std::ofstream ofs(filename);
  write_graphviz(ofs, dag, VertexLabelWriter(dag));
  ofs.close();
  std::cout << "DOT graph written to " << filename << std::endl;
}

// Helper to add a node if not already present
QuakeDAG::Vertex QuakeDAG::get_or_add_node(const std::string &name) {
  if (!node_map.contains(name)) {
    Vertex v = add_vertex(dag);
    dag[v].name = name;
    node_map[name] = v;
  }
  return node_map[name];
}