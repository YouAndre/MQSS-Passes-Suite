
/** @file
  @brief
  @details This header defines a set of functions that are useful to convert
  MLIR to DAG. DAG is useful to easily track the data dependencies.
  @par
  This header must be included to use the available functions to manipulate MLIR
  modules as DAGs.
*/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>

using namespace mlir;

struct MLIRVertex {
    std::string name;
    Operation *operation = nullptr; // quake operation
    func::FuncOp matrix;
    Value result;
    std::vector<int> targets;
    std::vector<int> controls;
    std::vector<double> arguments;
    bool isAdj = false;
    bool isQubit = false;
    bool isMeasurement = false;
};

// Define the graph type
class QuakeDAG {
public:
    QuakeDAG() = default;

    // Parses a Quake MLIR file to build the DAG
    void parse_mlir(func::FuncOp kernel);

    // Prints the DAG to the console
    void print() const;

    // Dumps the DAG to a .dot file for Graphviz/Dotty
    void dump_dot(const std::string &filename) const;

    using DAG = boost::adjacency_list<
        boost::vecS, boost::vecS, boost::bidirectionalS, MLIRVertex>;

    using Vertex = boost::graph_traits<DAG>::vertex_descriptor;
    using in_edge_iterator = boost::graph_traits<DAG>::in_edge_iterator;

    DAG &getGraph() { return dag; }
    const DAG &getGraph() const { return dag; }

private:
    DAG dag;
    std::unordered_map<std::string, Vertex> node_map;

    struct VertexLabelWriter {
        const DAG &g;

        explicit VertexLabelWriter(const DAG &graph) : g(graph) {
        }

        template<typename Vertex>
        void operator()(std::ostream &out, const Vertex &v) const {
            out << "[label=\"" << g[v].name << "\"]";
        }
    };

    // Adds a node to the graph if it doesn't exist
    Vertex get_or_add_node(const std::string &name);
};
