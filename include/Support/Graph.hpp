#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <unordered_set>
#include <algorithm>  // std::max
#include <iostream>
struct Node
{
    std::vector<Node*> parents;
    std::vector<Node*> children;
    uint32_t multiqubit = 0;
    uint32_t subjective_depth = 0;
};

class Graph
{
public:
    explicit Graph(uint32_t num_qubits);
    ~Graph();

    // Circuit DAG
    void create_node(const std::vector<int>& parents);

    // Interaction graph
    void create_interaction_node(int control, int target);

    // returns {depth, multiqubit_on_critical_path}
    std::pair<uint32_t, uint32_t> get_max_depths() const;

    // returns {program_communication, directed_program_communication}
    std::pair<double, double> compute_program_communication() const;

        // ---- DEBUG HELPERS ----
    void dump_nodes() const;
    void dump_degrees() const;
private:
    std::vector<Node*> nodes;         // owns all nodes
    std::vector<Node*> latest_record; // per-qubit leaves
};
