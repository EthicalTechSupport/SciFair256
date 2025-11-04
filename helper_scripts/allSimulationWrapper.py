# CREATED BY MANAS TELIKEPALLI

import argparse
import json
import sys
from pathlib import Path
import subprocess
import shutil

def run_all_simulations(num_loop,fasta_path, pdb_file = "ranked_0.pdb"):
    output_dir = Path.cwd() / "Execution" / "results" / str(num_loop)
    output_dir.mkdir(exist_ok=True)

    # ---------------- Pipeline Runners ----------------
    def run_seq2pdb_pipeline():
        result = subprocess.run(
            [sys.executable, "seqToPDB.py", fasta_path],
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("[Seq2PDB: Failure] Pipeline failed.")
        else:
            print("[Seq2PDB: Success] Pipeline executed successfully.")

    def run_kmc_pipeline():
        result = subprocess.run(
            [sys.executable, "./Dependencies_and_scripts/kmc/KMC_functions.py", fasta_path],
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("[KMC: Failure] Pipeline failed.")
        else:
            print("[KMC: Success] Pipeline executed successfully.")

    def run_docking_pipeline():
        result = subprocess.run(
            [sys.executable, "./Dependencies_and_scripts/mdock/MDock_functions.py", pdb_file],
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("[MDock: Failure] Pipeline failed.")
        else:
            print("[MDock: Success] Pipeline executed successfully.")

    def run_md_pipeline():
        result = subprocess.run(
            [sys.executable, "./Dependencies_and_scripts/mdynamics/MD_functions.py", pdb_file],
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("[MD: Failure] Pipeline failed.")
        else:
            print("[MD: Success] Pipeline executed successfully.")

    def collect_and_save_results():
        results = {
            "docking_score": 0.0,
            "gibbs_free_energy": 0.0,
            "avg_lipid_contacts": 0.0,
            "fibrils": 0,
            "longest_fibril": 0,
            "mass_frac": 0.0,
            "avg_fibril_length": 0.0,
            "nucleation": 0,
            "secondary": 0,
            "fragmentation": 0,
            "elongation": 0
        }

        # ---- Explicit targets ----
        move_files = [
            "dock_score.json",
            "best_pose.pdb",
            "lightdock.info",
            "pmf_analysis.txt",
            "avg_contacts.txt",
            "kmc_results.json",
            "histo.xvg",
            "pmf.xvg",
            "index_named.ndx",
            "protein.top",
            "protein_cg.gro",
            "pull.tpr",
            "ranked_0.pdb",
            "system_solvated.gro",
            "topol_updated.top"
        ]

        move_dirs = [
        ]

        delete_files = [
            "index.ndx",
            "pull.xtc",
            "protein_cg.pdb",
            "asyn_positioned.gro"
        ]

        delete_dirs = [
        ]

        # ---- Wildcard patterns ----
        move_patterns = [
            "kmc_out*"
        ]

        delete_patterns = [
            "npt_*",
            "*.dat",
            "umbrella_*",
            "*.bak",
            "conf*.gro"
        ]

        # ---- Helpers ----
        def move_file_if_exists(file_path):
            src = Path(file_path)
            if src.exists() and src.is_file():
                dest = output_dir / src.name
                try:
                    src.rename(dest)
                    print(f"[Move: Success] Moved file {src} to {dest}")
                    return dest
                except Exception as e:
                    print(f"[Move: Failure] Could not move file {src}: {e}")
            return None

        def move_dir_if_exists(dir_path):
            src = Path(dir_path)
            if src.exists() and src.is_dir():
                dest = output_dir / src.name
                try:
                    shutil.move(str(src), str(dest))
                    print(f"[Move: Success] Moved directory {src} to {dest}")
                    return dest
                except Exception as e:
                    print(f"[Move: Failure] Could not move directory {src}: {e}")
            return None

        def delete_file_if_exists(file_path):
            src = Path(file_path)
            if src.exists() and src.is_file():
                try:
                    src.unlink()
                    print(f"[Delete: Success] Deleted file {src}")
                    return src
                except Exception as e:
                    print(f"[Delete: Failure] Could not delete file {src}: {e}")
            return None

        def delete_dir_if_exists(dir_path):
            src = Path(dir_path)
            if src.exists() and src.is_dir():
                try:
                    shutil.rmtree(src)
                    print(f"[Delete: Success] Deleted directory {src}")
                    return src
                except Exception as e:
                    print(f"[Delete: Failure] Could not delete directory {src}: {e}")
            return None

        def move_pattern(pattern):
            moved = []
            for path in Path.cwd().glob(pattern):
                dest = output_dir / path.name
                try:
                    if path.is_dir():
                        shutil.move(str(path), str(dest))
                    else:
                        path.rename(dest)
                    moved.append(dest)
                except Exception as e:
                    print(f"[Move: Failure] Could not move {path}: {e}")
            if moved:
                print(f"[Move: Success] Moved {len(moved)} item(s) for pattern '{pattern}'.")
            return moved

        def delete_pattern(pattern):
            deleted = []
            for path in Path.cwd().glob(pattern):
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    deleted.append(path)
                except Exception as e:
                    print(f"[Delete: Failure] Could not delete {path}: {e}")
            if deleted:
                print(f"[Delete: Success] Deleted {len(deleted)} item(s) for pattern '{pattern}'.")
            return deleted

        # ---- Apply explicit lists ----
        for f in move_files:
            move_file_if_exists(f)
        for d in move_dirs:
            move_dir_if_exists(d)
        for f in delete_files:
            delete_file_if_exists(f)
        for d in delete_dirs:
            delete_dir_if_exists(d)

        # ---- Apply patterns ----
        for pattern in move_patterns:
            move_pattern(pattern)
        for pattern in delete_patterns:
            delete_pattern(pattern)

        # ---- Save results ----
        out_path = output_dir / "simulation_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"[Execution: Success] Final results written to: {out_path}")
        return results

    # ---------------- Main Control ----------------
    def main():
        try:
            print("[Execution: Start] Running simulation pipelines...")
            run_seq2pdb_pipeline()
            run_kmc_pipeline()
            run_docking_pipeline()
            run_md_pipeline()
            print("[Execution: Success] All pipelines completed.")
            final_results = collect_and_save_results()
            print(json.dumps(final_results, indent=4))
        except Exception as e:
            print(f"[Execution: Failure] Pipeline aborted due to error: {e}")
            sys.exit(1)

    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all simulations for a given loop index")
    parser.add_argument("--num_loop", required=True, help="Loop number for this simulation instance")
    parser.add_argument("--pdb_file", required=True, help="Path to mutant PDB file")
    parser.add_argument("--fasta_path", required=True, help="Path to mutant FASTA file")
    args = parser.parse_args()
    run_all_simulations(num_loop=args.num_loop, pdb_file=args.pdb_file, fasta_path=args.fasta_path)