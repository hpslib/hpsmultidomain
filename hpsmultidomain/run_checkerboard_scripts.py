import subprocess
from pathlib import Path

directory_name = "data-convection-helmholtz-kh100-halves"

# Define your directory path
dir_path = Path(directory_name)
dir_path.mkdir(parents=True, exist_ok=True)

p_list = [9, 11, 13, 15, 17, 19, 21]
#n_list = [36, 48, 60, 72]

for p in p_list:
    n_list = [8*(p-2), 24*(p-2), 72*(p-2), 216*(p-2)]
    for n in n_list:
        cmd = [
            "python", "hpsmultidomain/argparse_driver.py",
            "--pde", "convection_helmholtz_steady_state",
            "--domain", "square",
            "--bc", "convection_helmholtz_steady_state",
            "--n", str(n),
            "--p", str(p),
            "--d", "2",
            "--kh", "100",
            "--solver", "MUMPS",
            "--visualize", "True",
            "--pickle", directory_name + f"/helmholtz-p{p}-n{n}.pkl",
            "--store_sol",
        ]
        print(f"Running p={p}, n={n}...")
        subprocess.run(cmd, check=True)

print("All runs complete.")