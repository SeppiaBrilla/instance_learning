import random
import subprocess
import os
from tqdm import tqdm

def generate_instance_matrix(n, min_dist, max_dist):
    dist = [["0" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = str(random.randint(min_dist, max_dist))
            dist[j][i] = dist[i][j]
    return dist
    
def generate_opt_instance_dzn(n, min_dist, max_dist, dist, name):
    dist_lines = [", ".join(d) for d in dist]
    with open(f"./.cache/{name}.dzn", "w") as f:
        f.write(f"n = {n};\n")
        f.write(f"dist = [| {'|\n'.join(dist_lines)} |];\n")
        f.write(f"min_dist = {min_dist};\n")
        f.write(f"max_dist = {max_dist};\n")

def solve_opt_instance_dzn(name):
    res = subprocess.run(["minizinc", "./data/mzn_models/tsp_opt.mzn", f"./.cache/{name}.dzn", "--solver", "cp-sat"], stdout=subprocess.PIPE)
    full_res = res.stdout.decode("utf-8")
    n_res = full_res.split("----------")[0].replace("Cost: ", "").strip()
    return int(n_res)

def generate_sat_dist(n, min_dist, max_dist, dist, name, sol):
    dist_lines = [", ".join(d) for d in dist]
    with open(f"./tsp_instances/{name}.dzn", "w") as f:
        f.write(f"n = {n};\n")
        f.write(f"dist = [| {'|\n'.join(dist_lines)} |];\n")
        f.write(f"min_dist = {min_dist};\n")
        f.write(f"max_dist = {max_dist};\n")
        f.write(f"sol = {sol};\n")
        

def generate_graph(i:int, sat:bool, n:int, min_dist:int, max_dist:int):
    stdout = subprocess.run([f'minizinc ./data/mzn_models/tsp_sat.mzn ./data/tsp_instances/tsp_{i}.dzn --solver gecode -c --no-output-ozn --fzn .cache/tsp_{i}.fzn'], shell=True, stderr=subprocess.PIPE).stderr.decode()
    if "Warning: model inconsistency detected" in stdout:
        return False
    subprocess.run([f'python ./src/flatzinc_parser/flatzinc_parser.py .cache/tsp_{i}.fzn ./data/tsp_graphs/tsp_{i}.graph'], shell=True, stdout=subprocess.DEVNULL)
    subprocess.run([f'mv .cache/tsp_{i}.fzn ./data/tsp_flat/tsp_{i}.fzn'], shell=True)
    with open(f"./data/tsp_graphs/tsp_{i}.graph", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"%n: {n}, min_dist: {min_dist}, max_dist: {max_dist}, sat: {str(sat).lower()}" + '\n' + content)
    return True



Ns = [5, 6, 7, 8, 9, 10]
Maxes = [10, 15, 20]
Mins = [1, 2, 4]

num_instances = 10000

for i in tqdm(range(num_instances)):
    generated = False
    if os.path.exists(f"../data/tsp_graphs/tsp_{i}.graph"):
        generated = True
    while not generated:
        n = random.choice(Ns)
        max_d = random.choice(Maxes)
        min_d = random.choice(Mins)
        dist_matrix = generate_instance_matrix(n, min_d, max_d)
        generate_opt_instance_dzn(n, min_d, max_d, dist_matrix, f"opt_inst_{i}")
        sol = solve_opt_instance_dzn(f"opt_inst_{i}")
        sat = i % 2 == 0
        if sat:
            generate_sat_dist(n, min_d, max_d, dist_matrix, f"tsp_{i}", int(sol + sol*0.1))
            generated = generate_graph(i, sat, n, min_d, max_d)
        else:
            generate_sat_dist(n, min_d, max_d, dist_matrix, f"tsp_{i}", int(sol - sol*0.01))
            generated = generate_graph(i, sat, n, min_d, max_d)
