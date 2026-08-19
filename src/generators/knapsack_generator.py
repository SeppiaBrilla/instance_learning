import random
from uuid import uuid4
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

NS = [20, 25, 30, 35, 40, 45, 50]

CAPACITY = [20, 40, 60, 80, 100]

MAX_PROFIT = [10, 20, 30, 40, 50]
MAX_SIZE = [30, 50, 70, 90, 110]

def get_profits(rng, n, max_profit) -> list[int]:
    return [rng.randint(1, max_profit) for _ in range(n)]

def get_sizes(rng, n, max_size) -> list[int]:
    return [rng.randint(1, max_size) for _ in range(n)]

def get_random_optim_instance(rng) -> dict:
    n = rng.choice(NS)
    max_profit = rng.choice(MAX_PROFIT)
    max_size = rng.choice(MAX_SIZE)
    capacity = rng.choice(CAPACITY)
    profits = get_profits(rng, n, max_profit)
    sizes = get_sizes(rng, n, max_size)
    return {
        "n": n,
        "capacity": capacity,
        "profits": profits,
        "sizes": sizes
    }

def save_opt(instance, filename) -> None:
    with open(filename, "w") as f:
        f.write(f"""n = {instance['n']};
capacity = {instance['capacity']};
profit = {instance['profits']};
size = {instance['sizes']};""")

def save_sat(instance, filename) -> None:
    with open(filename, "w") as f:
        f.write(f"""n = {instance['n']};
capacity = {instance['capacity']};
profit = {instance['profits']};
size = {instance['sizes']};
desired_profit = {instance['max_profit']};""")

def get_max_profit(instance):
    res = subprocess.run([f'minizinc ./data/mzn_models/model_knapsack_opt.mzn {instance} --solver cp-sat'], stdout=subprocess.PIPE, shell=True)
    res_txt = res.stdout.decode('utf-8')
    proc_line = res_txt.splitlines()[0]
    return int(proc_line.split("p = ")[1]) 


def generate_instance(rng, sat:bool, i, name:str) -> dict:
    instance = get_random_optim_instance(rng)
    inst_path = f'./.cache/instance_{i}.dzn'
    save_opt(instance, inst_path)
    max_profit = get_max_profit(inst_path)
    if sat:
        instance['max_profit'] = int(max_profit * 0.98)
        save_sat(instance, name)
    else:
        instance['max_profit'] = int(max_profit * 1.02)
        save_sat(instance, name)
    return instance

def generate_graph(i:int, sat:bool, capacity:int, profit, size:int, desired_profit:int):
    stdout = subprocess.run([f'minizinc ./data/mzn_models/model_knapsack_sat.mzn ./data/knapsack_instances/instance_{i}.dzn --solver gecode -c --no-output-ozn --fzn .cache/instance_knapsack_{i}.fzn'], shell=True, stderr=subprocess.PIPE).stderr.decode()
    if "Warning: model inconsistency detected" in stdout:
        raise Exception(f"Model inconsistency detected {i}")
    subprocess.run([f'python ./flatzinc_parser/flatzinc_parser.py .cache/instance_knapsack_{i}.fzn ./data/knapsack_graphs/instance_knapsack_{i}.graph'], shell=True)
    subprocess.run([f'mv .cache/instance_knapsack_{i}.fzn ./data/knapsack_flat/instance_knapsack_{i}.fzn'], shell=True)
    with open(f"./data/knapsack_graphs/instance_knapsack_{i}.graph", 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"%capacity: {capacity}, profit: {profit}, size: {size}, desired_profit: {desired_profit}, sat: {str(sat).lower()}" + '\n' + content)


def process_instance(i):
    rng = random.Random(i)
    instance = generate_instance(rng, bool(i%2), i, name=f"knapsack_instances/instance_{i}.dzn")
    generate_graph(i, bool(i%2), instance['capacity'], instance['profits'], instance['sizes'], instance['max_profit'])
    return i

def main():
    existing = 0 #int(subprocess.run(["ls knapsack_instances/ | wc -l"], shell=True, stdout=subprocess.PIPE).stdout.decode().strip())
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_instance, i): i
            for i in range(existing, 10000)
        }
        for future in as_completed(futures):
            i = futures[future]
            future.result()

if __name__ == "__main__":
    main()