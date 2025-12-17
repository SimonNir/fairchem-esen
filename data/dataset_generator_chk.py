import os
import glob
import h5py
from pyscf import lib

chk_dirs = [
    "/scratch/aaldo/egsmole/test_dataset/amino_acids",
    "/scratch/aaldo/egsmole/test_dataset/alcohols",
    "/scratch/aaldo/egsmole/test_dataset/alkanes",
    "/scratch/aaldo/egsmole/test_dataset/pubchem",
    "/scratch/aaldo/egsmole/test_dataset/stretching_mol",
    "/scratch/aaldo/egsmole/qm7",
]

atomic_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

def read_data(chk_path):
    mol = lib.chkfile.load_mol(chk_path)
    atoms = mol.atom  # list of (symbol, (x, y, z))
    
    try:
        with h5py.File(chk_path, "r") as f:
            e_mp2 = f["mp2/MP2/e_corr"][()]
            e_ccsd = f["ccsd/e_corr"][()]

    except Exception as e:
        print(f"Error reading {chk_path}: {e}")
        return None, None

    # in case we are dealing with atomic numbers, convert to symbols
    coords = mol.atom_coords(unit='Angstrom')
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    atoms = list(zip(symbols, coords))
    symbols = [sym for sym, (x, y, z) in atoms]
    if all(sym in atomic_dict.keys() for sym in symbols):
        atoms = [(atomic_dict[sym], (x, y, z)) for sym, (x, y, z) in atoms]

    return atoms, e_ccsd - e_mp2

for chk_dir in chk_dirs:
    output_file = os.path.basename(chk_dir) + ".xyz"
    chk_path_list = glob.glob(os.path.join(chk_dir, "*.chk"))
    chk_path_list_sorted = sorted(chk_path_list, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))  # sort by molecule ID
    with open(output_file, "w") as out:
        for chk_path in chk_path_list_sorted:
            # print(f"Writing to {chk_path}")
            atoms, delta_e = read_data(chk_path)
            if atoms is None:
                print(f"Skipping {chk_path} due to read error")
                continue
            # print(f"{atoms}, delta_e={delta_e:.6f}")
            out.write(f"{len(atoms)}\nProperties=species:S:1:pos:R:3 REF_energy={delta_e}\n")
            for sym, (x, y, z) in atoms:
                out.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")