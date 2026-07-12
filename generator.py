import random
import networkx as nx
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

num_vertices = [10, 12, 14, 16, 18, 20]
edge_probability = [.2, .4, .6, .8]
num_colors = [3, 5, 7, 9, 11, 13, 15]

def generate_coloring_instance(num_vertices, edge_probability, num_colors, filename="instance.dzn"):
    """
    Generates a random graph and writes it as a MiniZinc .dzn data file.
    """
    # Generate a random Erdős-Rényi graph
    graph = nx.erdos_renyi_graph(n=num_vertices, p=edge_probability)
    # Extract the edges and convert 0-indexed nodes to 1-indexed for MiniZinc
    edges = [(u + 1, v + 1) for u, v in graph.edges()]
    num_edges = len(edges)

    # Format the edges array into MiniZinc 2D array syntax [| u1, v1 | u2, v2 | ... |]
    if num_edges > 0:
        edge_strings = [f" {u}, {v} " for u, v in edges]
        minizinc_edges = "| " + " | ".join(edge_strings) + " |"
    else:
        minizinc_edges = ""

    # Write the .dzn file
    with open(filename, "w") as f:
        f.write(f"% Generated Graph Coloring Instance\n")
        f.write(f"nc = {num_colors};\n")
        f.write(f"nv = {num_vertices};\n")
        f.write(f"ne = {num_edges};\n\n")
        f.write(f"edges = [{minizinc_edges}];\n")

def generate_graph(i, num_vertices, edge_probability, num_colors):
    stdout = subprocess.run([f'minizinc model.mzn instances/instance_{i}.dzn --solver cp-sat'], shell=True, stdout=subprocess.PIPE).stdout.decode()
    sat =  not "=====UNSATISFIABLE=====" in stdout
    
    subprocess.run([f'minizinc model.mzn instances/instance_{i}.dzn --solver gecode -c --no-output-ozn --fzn .cache/instance_{i}.fzn'], shell=True)
    subprocess.run([f'python ../flatzinc_parser/flatzinc_parser.py .cache/instance_{i}.fzn graphs/instance_{i}.graph'], shell=True)
    subprocess.run([f'rm .cache/instance_{i}.fzn'], shell=True)
    with open(f"graphs/instance_{i}.graph", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"%nc: {num_colors}, nv: {num_vertices}, sat: {str(sat).lower()}" + '\n' + content)

def process_instance(i, num_vertices, edge_probability, num_colors):
    # Use a per-task RNG so parallel workers don't share/clash random state
    rng = random.Random(i)
    n_vertices = rng.choice(num_vertices)
    e_prob = rng.choice(edge_probability)
    n_cols = rng.choice(num_colors)

    generate_coloring_instance(num_vertices=n_vertices,
                               edge_probability=e_prob,
                               num_colors=n_cols,
                               filename=f"instances/instance_{i}.dzn")
    generate_graph(i, n_vertices, e_prob, n_cols)
    return i

def main():
    existing = int(subprocess.run(["ls instances/ | wc -l"], shell=True, stdout=subprocess.PIPE).stdout.decode().strip())
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_instance, i, num_vertices, edge_probability, num_colors): i
            for i in range(existing, 10000)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Instance {i} failed: {e}")

if __name__ == "__main__":
    main()
