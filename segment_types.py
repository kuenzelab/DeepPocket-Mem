import argparse
import os
import glob
import numpy as np
from openbabel import pybel


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Convert classification .types (label x y z gninatypes) to "
            "segmentation .types (label x y z gninatypes ligand), "
            "handling multiple ligand files per protein."
        )
    )
    p.add_argument("--train_types", required=True, help="classification train.types")
    p.add_argument("--test_types",  required=True, help="classification test.types")
    p.add_argument(
        "--ligand_root", default="../ligands",
        help="root folder with ligand files like PDBCODE_lig.pdb, PDBCODE_lig2.pdb, ..."
    )
    p.add_argument(
        "--out_train", default="seg_train.types",
        help="output segmentation train.types"
    )
    p.add_argument(
        "--out_test",  default="seg_test.types",
        help="output segmentation test.types"
    )
    p.add_argument(
        "--use_only_positives", action="store_true",
        help="keep only label=1 entries (recommended for seg training)"
    )
    p.add_argument(
        "--max_cutoff_to_incl_lig", type=float, default=10.0,
        help="max allowed centre-to-ligand distance (Å); centres farther than this are skipped"
    )
    return p.parse_args()




_ligand_com_cache = {}  # ligand_path -> list of COMs (one per TER-block)


def compute_ligand_coms(ligand_path):
    if ligand_path in _ligand_com_cache:
        return _ligand_com_cache[ligand_path]

    if not os.path.isfile(ligand_path):
        raise FileNotFoundError(f"Ligand file not found: {ligand_path}")

    with open(ligand_path, "r") as f:
        lines = f.read().split("\n")

    current_ligand = []
    coms = []

    def _coords_from_block(block_lines):
        mol = pybel.readstring("pdb", "\n".join(block_lines))
        coords = np.array([atom.coords for atom in mol.atoms], dtype=float)
        if coords.size == 0:
            return None
        return coords

    for line in lines:
        if line.startswith("HETATM") or line.startswith("ATOM"):
            current_ligand.append(line)
        elif line.startswith("TER") and current_ligand:
            coords = _coords_from_block(current_ligand)
            if coords is not None:
                coms.append(coords.mean(axis=0))
            current_ligand = []

    
    if current_ligand:
        coords = _coords_from_block(current_ligand)
        if coords is not None:
            coms.append(coords.mean(axis=0))

    if not coms:
        raise RuntimeError(f"No ligand atoms found in {ligand_path}")

    _ligand_com_cache[ligand_path] = coms
    return coms


def get_pdbcode_from_gninatypes(gninatypes_path):
    
    base = os.path.basename(gninatypes_path)   
    stem, _ = os.path.splitext(base)         
    pdbcode = stem.replace("_nowat", "")     
    return pdbcode


def find_ligand_candidates(pdbcode, ligand_root):
    pattern = os.path.join(ligand_root, f"{pdbcode}_lig*.pdb")
    return sorted(glob.glob(pattern))


def choose_best_ligand_for_center(center_xyz, ligand_paths):
    center = np.array(center_xyz, dtype=float)
    best_path = None
    best_dist = None

    for lig_path in ligand_paths:
        try:
            coms = compute_ligand_coms(lig_path)
        except Exception as e:
            print(f"[WARN] Skipping {lig_path}: {e}")
            continue

        # distance to closest COM in that ligand file
        dists = [np.linalg.norm(center - com) for com in coms]
        if not dists:
            continue
        dmin = min(dists)

        if best_dist is None or dmin < best_dist:
            best_dist = dmin
            best_path = lig_path

    return best_path, best_dist


def convert_one(in_types, out_types, ligand_root,
                use_only_positives=True, max_cutoff_to_incl_lig=10.0):
    n_in, n_out, n_skipped_no_lig, n_skipped_far = 0, 0, 0, 0

    with open(in_types, "r") as fin, open(out_types, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] skipping malformed line: {line}")
                continue

            n_in += 1
            label = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            rec_path = parts[4]

            if use_only_positives and label != 1:
                continue  

            pdbcode = get_pdbcode_from_gninatypes(rec_path)
            lig_candidates = find_ligand_candidates(pdbcode, ligand_root)

            if not lig_candidates:
                n_skipped_no_lig += 1
                print(f"[WARN] No ligand files found for {pdbcode} in {ligand_root}")
                continue

            best_lig, best_dist = choose_best_ligand_for_center((x, y, z), lig_candidates)
            if best_lig is None:
                n_skipped_no_lig += 1
                continue

            if best_dist is None or best_dist > max_cutoff_to_incl_lig:
                n_skipped_far += 1
                continue

            # segmentation label can just be 1 (we're training only positives)
            fout.write(f"1 {x} {y} {z} {rec_path} {best_lig}\n")
            n_out += 1

    if n_skipped_no_lig:
        print(f"[INFO]  skipped {n_skipped_no_lig} centres with no ligand files found")
    if n_skipped_far:
        print(f"[INFO]  skipped {n_skipped_far} centres with best ligand farther than max_cutoff_to_incl_lig={max_cutoff_to_incl_lig} Å")


def main():
    args = parse_args()
    convert_one(
        args.train_types, args.out_train,
        args.ligand_root,
        use_only_positives=args.use_only_positives,
        max_cutoff_to_incl_lig=args.max_cutoff_to_incl_lig,
    )
    convert_one(
        args.test_types, args.out_test,
        args.ligand_root,
        use_only_positives=args.use_only_positives,
        max_cutoff_to_incl_lig=args.max_cutoff_to_incl_lig,
    )


if __name__ == "__main__":
    main()
