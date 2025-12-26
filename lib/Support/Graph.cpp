#include "Support/Graph.hpp"




Graph::Graph(uint32_t num_qubits)
    : latest_record(num_qubits, nullptr)
{
    // Initialize one root node per qubit
    for (uint32_t q = 0; q < num_qubits; ++q) {
        Node* n = new Node();
        nodes.push_back(n);
        latest_record[q] = n;
    }
}

Graph::~Graph()
{
    for (Node* n : nodes) {
        delete n;
    }
}
void Graph::create_interaction_node(const int control, const int target)
{
    // Defensive checks
    if (control < 0 || target < 0 ||
        control >= static_cast<int>(latest_record.size()) ||
        target >= static_cast<int>(latest_record.size()))
        return;

    Node* c = latest_record[control];
    Node* t = latest_record[target];

    if (!c || !t || c == t)
        return;

    // ----- Directed edge: c -> t -----

    // Add t as child of c if not already present
    auto& cChildren = c->children;
    if (std::find(cChildren.begin(), cChildren.end(), t) == cChildren.end()) {
        cChildren.push_back(t);
    }

    // Add c as parent of t if not already present
    auto& tParents = t->parents;
    if (std::find(tParents.begin(), tParents.end(), c) == tParents.end()) {
        tParents.push_back(c);
    }
}
std::pair<double, double>
Graph::compute_program_communication() const
{
    const uint32_t num_qubits = latest_record.size();
    if (num_qubits < 1)
        return {0.0, 0.0};
    if (num_qubits < 2)
        return {1.0, 0.0};

    uint64_t degree_sum = 0;
    uint64_t degree_sum2 = 0;

    for (Node* n : latest_record) {
        if (!n)
            continue;

        // ----- directed degree -----
        const uint32_t in_deg  = n->parents.size();
        const uint32_t out_deg = n->children.size();
        degree_sum2 += in_deg + out_deg;

        // ----- undirected degree -----
        // unique parents ∪ children
        std::unordered_set<Node*> uniq;
        uniq.reserve(in_deg + out_deg);

        for (Node* p : n->parents)
            uniq.insert(p);
        for (Node* c : n->children)
            uniq.insert(c);

        degree_sum += uniq.size();
    }

    const double max_degree =
        static_cast<double>(num_qubits) *
        static_cast<double>(num_qubits - 1);

    const double program_communication =
        max_degree > 0.0
            ? static_cast<double>(degree_sum/2) / max_degree
            : 0.0;

    const double directed_program_communication =
        max_degree > 0.0
            ? static_cast<double>(degree_sum2) / (2.0 * max_degree)
            : 0.0;

    return {program_communication, directed_program_communication};
}

void Graph::create_node(const std::vector<int>& parents)
{
    Node* n = new Node();
    nodes.push_back(n);

    uint32_t max_depth = 0;
    uint32_t inherited_multiqubit = 0;

    // Attach parents and find critical parent
    for (int q : parents) {
        Node* p = latest_record[q];
        if (!p) continue;

        n->parents.push_back(p);
        p->children.push_back(n);

        if (p->subjective_depth >= max_depth) {
            max_depth = p->subjective_depth;
            inherited_multiqubit = p->multiqubit;
        }
    }

    // Update depth
    n->subjective_depth = max_depth + 1;

    // Update multiqubit depth along critical path
    if (parents.size() > 1) {
        n->multiqubit = inherited_multiqubit + 1;
    } else {
        n->multiqubit = inherited_multiqubit;
    }

    // Update per-qubit latest record
    for (int q : parents) {
        latest_record[q] = n;
    }
}

std::pair<uint32_t, uint32_t> Graph::get_max_depths() const
{
    uint32_t max_depth = 0;
    uint32_t multiqubit_at_max_depth = 0;

    for (Node* n : latest_record) {
        if (!n) continue;

        if (n->subjective_depth > max_depth) {
            max_depth = n->subjective_depth;
            multiqubit_at_max_depth = n->multiqubit;
        }
        else if (n->subjective_depth == max_depth) {
            multiqubit_at_max_depth =
                std::max(multiqubit_at_max_depth, n->multiqubit);
        }
    }

    return {max_depth, multiqubit_at_max_depth};
}
void Graph::dump_nodes() const
{
    std::cout << "=== GRAPH NODES ===\n";
    std::cout << "Total nodes: " << nodes.size() << "\n\n";

    std::unordered_map<const Node*, uint32_t> node_ids;
    for (uint32_t i = 0; i < nodes.size(); ++i)
        node_ids[nodes[i]] = i;

    for (uint32_t i = 0; i < nodes.size(); ++i) {
        const Node* n = nodes[i];

        std::cout << "Node " << i
                  << " | depth=" << n->subjective_depth
                  << " | multiqubit=" << n->multiqubit
                  << "\n";

        std::cout << "  Parents: ";
        for (const Node* p : n->parents)
            std::cout << node_ids[p] << " ";
        std::cout << "\n";

        std::cout << "  Children: ";
        for (const Node* c : n->children)
            std::cout << node_ids[c] << " ";
        std::cout << "\n\n";
    }

    std::cout << "Latest record per qubit:\n";
    for (uint32_t q = 0; q < latest_record.size(); ++q) {
        std::cout << "  q" << q << " -> node "
                  << node_ids.at(latest_record[q]) << "\n";
    }

    std::cout << "====================\n\n";
}
void Graph::dump_degrees() const
{
    std::cout << "=== GRAPH DEGREES ===\n";

    uint64_t degree_sum = 0;
    uint64_t degree_sum2 = 0;

    for (uint32_t q = 0; q < latest_record.size(); ++q) {
        const Node* n = latest_record[q];
        if (!n)
            continue;

        const uint32_t in_deg  = n->parents.size();
        const uint32_t out_deg = n->children.size();

        std::unordered_set<const Node*> uniq;
        for (const Node* p : n->parents)
            uniq.insert(p);
        for (const Node* c : n->children)
            uniq.insert(c);

        std::cout << "Qubit " << q
                  << " | in=" << in_deg
                  << " | out=" << out_deg
                  << " | undirected=" << uniq.size()
                  << "\n";

        degree_sum  += uniq.size();
        degree_sum2 += in_deg + out_deg;
    }

    const uint32_t num_qubits = latest_record.size();
    const double max_degree =
        static_cast<double>(num_qubits) *
        static_cast<double>(num_qubits - 1);

    std::cout << "\nDegree sum (undirected): " << degree_sum/2.0 << "\n";
    std::cout << "Degree sum (directed):   " << degree_sum2/2.0 << "\n";
    std::cout << "Max degree:              " << max_degree << "\n";

    if (max_degree > 0.0) {
        std::cout << "Program communication: "
                  << (degree_sum / 2.0) / max_degree << "\n";
        std::cout << "Directed communication: "
                  << degree_sum2 / (2.0 * max_degree) << "\n";
    }

    std::cout << "======================\n\n";
}
