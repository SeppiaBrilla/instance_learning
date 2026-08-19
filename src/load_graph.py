import re
import numpy as np

node_types = ['mult_node', 'literal_node', 'inequality_node', 'lin_sum_node', 'par_node', 'var_node', 'equality_node', 'leq_node', 'geq_node', 'circuit_node', 'int_element_node']

def one_hot_encode_type_fn(node_type:str) -> np.ndarray:
    ohe = np.array([1 if node_type == nt else 0 for nt in node_types])
    return ohe

def load_graph_to_features(file_path):
    """Parses the graph file and extracts node and edge features for an encoder.

    Node features array columns: [one_hot_types..., label_type, lb, ub]
    """
    nodes = {}
    edges = []  # Elements will be tuples: (src, dst, edge_type)

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse sections
    current_section = None
    for line in lines:
        if line == "nodes:":
            current_section = "nodes"
            continue
        elif line == "edges:":
            current_section = "edges"
            continue
        elif line.startswith("%"):
            continue

        if current_section == "nodes":
            # Format: 'idx: label -- type -- extra'
            parts = [p.strip() for p in line.split("--")]
            first_part = parts[0]
            idx_str, _ = first_part.split(":", 1)
            node_idx = int(idx_str.strip())
            node_type = parts[1].strip()

            # Initialize defaults for features
            # label_type: 0 for normal/other, 1 for literal_node/par_node
            label_type = 0.0
            lb = 0.0
            ub = 0.0

            if node_type in ("literal_node", "par_node"):
                label_type = 1.0

            elif node_type == "var_node" and len(parts) > 2:
                domain = parts[2].strip()
                match = re.match(r"(-?\d+)\.\.(-?\d+)", domain)
                if match:
                    lb = float(match.group(1))
                    ub = float(match.group(2))

            type_one_hot = one_hot_encode_type_fn(node_type)
            feature_vector = list(type_one_hot) + [label_type, lb, ub]
            nodes[node_idx] = feature_vector

        elif current_section == "edges":
            # New Format: edge_idx: idx1--idx2--edge_type
            _, connection_part = line.split(":", 1)
            parts = [p.strip() for p in connection_part.split("--")]
            
            src = int(parts[0])
            dst = int(parts[1])
            edge_type = int(parts[2])  # 0 or 1
            
            edges.append((src, dst, edge_type))

    # Map node indices to a contiguous range [0, num_nodes - 1]
    sorted_node_indices = sorted(nodes.keys())
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_node_indices)}

    # Construct the continuous node feature matrix
    x = [nodes[old_idx] for old_idx in sorted_node_indices]

    # Map edge indices and separate structural connectivity from edge attributes
    mapped_edges = []
    edge_attr = []
    
    for src, dst, edge_type in edges:
        if src in idx_map and dst in idx_map:
            mapped_edges.append([idx_map[src], idx_map[dst]])
            edge_attr.append([edge_type])

    # Transpose mapped_edges into [2, num_edges] matrix
    edge_index = list(map(list, zip(*mapped_edges))) if mapped_edges else [[], []]

    return {
        "x": x,                 # Node feature matrix
        "edge_index": edge_index,  # Adjacency list format [Source, Destination]
        "edge_attr": edge_attr   # Edge type feature matrix
    }

if __name__ == '__main__':
    res = load_graph_to_features("graphs/instance_100.graph")
    print(res["x"])
    print(res["edge_index"])
    print(res["edge_attr"])
