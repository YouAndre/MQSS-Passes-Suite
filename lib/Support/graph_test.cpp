#include "Support/Graph.hpp"


#include <iostream>
#include <cassert>

int main() {
    // ---- Create a 3-qubit graph ----
    Graph g(3);
    Graph g2(3);

    // Single-qubit gates
    g.create_node({0});  // q0
    g.create_node({1});  // q1
    g.create_node({2});  // q2

    // Two-qubit interaction: CX 0 -> 1
    g.create_node({0, 1});
    g2.create_interaction_node(0, 1);

    // Another interaction: CX 1 -> 2
    g.create_node({1, 2});
    g2.create_interaction_node(1, 2);
    // A three-qubit gate (for testing multiqubit depth)
    g.create_node({1, 0});
    g2.create_interaction_node(1, 0);
    g.create_node({2, 1});
    g2.create_interaction_node(2, 1);
    
    
    // ---- Check depths ----
    auto [depth, multiqubit] = g.get_max_depths();

    std::cout << "Max depth: " << depth << "\n";
    std::cout << "Multiqubit depth: " << multiqubit << "\n";

    assert(depth > 0);
    assert(multiqubit == 4);

    // ---- Check communication metrics ----
    std::cout << "\n--- CIRCUIT GRAPH ---\n";
    g.dump_nodes();
    g.dump_degrees();

    std::cout << "\n--- INTERACTION GRAPH ---\n";
    g2.dump_nodes();
    g2.dump_degrees();
    auto [pc, dpc] = g2.compute_program_communication();

    std::cout << "Program communication: " << pc << "\n";
    std::cout << "Directed program communication: " << dpc << "\n";

    assert(pc > 0.0);
    assert(dpc > 0.0);

    std::cout << "Graph test PASSED\n";
    return 0;
}
