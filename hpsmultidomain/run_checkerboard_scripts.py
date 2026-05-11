import subprocess
from pathlib import Path

kh        = 50
b         = 0
checkered = False
shifted   = False

directory = "data-helmholtz-"

if checkered:
    directory = directory + "checkerboard-"
    if shifted:
        directory = directory + "shifted-"

directory_name = directory + "kh" + str(kh) + "-b" + str(b)

# Define your directory path
dir_path = Path(directory_name)
dir_path.mkdir(parents=True, exist_ok=True)

#p_list = [9, 11, 13, 15, 17, 19, 21]
#p_list = [4, 6, 8, 10, 12, 14, 16, 18]

p_list = [5, 7, 9, 11, 13, 15, 17]

pde = "helmholtz"
if checkered:
    pde = pde + "_checkerboard"

for p in p_list:
    n_list = [2*(p-2), 4*(p-2), 8*(p-2), 16*(p-2), 32*(p-2), 64*(p-2), 128*(p-2)]
    #n_list = [2*(p-2), 6*(p-2), 18*(p-2), 54*(p-2), 162*(p-2)]
    #n_list = [8*(p-2), 24*(p-2), 72*(p-2), 216*(p-2)]
    #n_list = [256*(p-2)]
    for n in n_list:
        cmd = [
            "python", "hpsmultidomain/argparse_driver.py",
            "--pde", pde,
            "--domain", "square",
            "--bc", "convection_helmholtz_steady_state",
            "--n", str(n),
            "--p", str(p),
            "--d", "2",
            "--kh", str(kh),
            "--solver", "MUMPS",
            "--visualize", directory_name,
            "--pickle", directory_name + f"/helmholtz-p{p}-n{n}.pkl",
            "--store_sol",
        ]
        print(f"Running p={p}, n={n}...")
        subprocess.run(cmd, check=True)

print("All runs complete.")