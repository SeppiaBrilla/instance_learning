import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_instance(instance):
    # Determine file paths
    fzn_filename = instance.replace("dzn", "fzn")
    fzn_path = f"flat/{fzn_filename}"

    # Skip if already exists
    if os.path.exists(fzn_path):
        return

    # Run MiniZinc command
    cmd = f"minizinc -c --no-output-ozn --solver gecode model.mzn instances/{instance} --fzn {fzn_path}"
    subprocess.run([cmd], shell=True)

    # Filter out comments from the generated file
    if os.path.exists(fzn_path):
        lines = []
        with open(fzn_path, "r") as f:
            for line in f:
                if line[0] != "%":
                    if "%" in line:
                        lines.append(line[: line.index("%")])
                    else:
                        lines.append(line)

        with open(fzn_path, "w") as f:
            f.write("\n".join(lines))


def main():
    instance_dir = "./instances/"
    if not os.path.exists(instance_dir):
        print(f"Directory {instance_dir} does not exist.")
        return

    instances = os.listdir(instance_dir)

    # Execute over 10 cores using a process pool
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_instance, instance): instance
            for instance in instances
        }

        # Track progress with tqdm as processes complete
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            pass


if __name__ == "__main__":
    main()
