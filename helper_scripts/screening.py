import subprocess
import os
import json
from pathlib import Path
import sys

# Add AIUPred local installation path
sys.path.append(r"C:\Users\mteli\AIUPred")
import aiupred_lib


def run_tango_from_fasta(fasta_file, tango_exe, work_dir, output_json,
                         ph=7.4, te=303, io=0.05, tf=0.0, ct="N", nt="N"):
    """
    Run TANGO on sequences from a FASTA file, parse residue-level outputs,
    save results to JSON, and clean up intermediate .txt files.

    Returns a dict: {seq_id: {"avg_agg": float, "residues": [...]}}
    """

    # --- 1. Read FASTA ---
    sequences = {}
    with open(fasta_file, 'r') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = ''.join(seq_lines)
                seq_id = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id:
            sequences[seq_id] = ''.join(seq_lines)

    print(f"[DEBUG] Parsed sequences: {list(sequences.keys())}")

    results = {}

    # --- 2. Run TANGO ---
    for seq_id, seq in sequences.items():
        cmd = [
            tango_exe,
            seq_id,
            f'ct="{ct}"',
            f'nt="{nt}"',
            f'ph="{ph}"',
            f'te="{te}"',
            f'io="{io}"',
            f'tf="{tf}"',
            f'seq="{seq}"'
        ]
        cmd_str = " ".join(cmd)
        print(f"[DEBUG] CMD: {cmd_str}")

        try:
            completed = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                shell=True
            )
            print(f"[DEBUG] TANGO stdout for {seq_id}:\n{completed.stdout}")
            if completed.stderr.strip():
                print(f"[DEBUG] TANGO stderr for {seq_id}:\n{completed.stderr}")
        except Exception as e:
            print(f"[ERROR] Failed to run TANGO for {seq_id}: {e}")
            results[seq_id] = None
            continue

        # --- 3. Parse residue-level output ---
        out_file = os.path.join(work_dir, f"{seq_id}.txt")
        residues = []
        if os.path.isfile(out_file):
            with open(out_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("res") or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        res_dict = {
                            "res": int(parts[0]),
                            "aa": parts[1],
                            "Beta": float(parts[2]),
                            "Turn": float(parts[3]),
                            "Helix": float(parts[4]),
                            "Aggregation": float(parts[5]),
                            "HelicalAgg": float(parts[6]) if len(parts) > 6 else 0.0
                        }
                        residues.append(res_dict)
                    except ValueError:
                        print(f"[DEBUG] Could not parse line: {line}")
                        continue

        # --- 4. Compute average aggregation ---
        avg_agg = None
        if residues:
            avg_agg = sum(r["Aggregation"] for r in residues) / len(residues)
        results[seq_id] = {"avg_agg": avg_agg, "residues": residues}

    # --- 5. Save JSON ---
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] TANGO results saved to {output_json}")

    # --- 6. Cleanup .txt files ---
    for seq_id in sequences.keys():
        txt_file = os.path.join(work_dir, f"{seq_id}.txt")
        if os.path.exists(txt_file):
            os.remove(txt_file)
            print(f"[DEBUG] Deleted intermediate file: {txt_file}")

    return results


def run_aiupred_local(fasta_file, output_json, analysis_type="disorder"):
    """
    Run AIUPred locally on a multifasta file and save results to a JSON file.
    """
    fasta_path = Path(fasta_file)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read sequences from multifasta
    with open(fasta_path) as fh:
        sequences = aiupred_lib.multifasta_reader(fh)

    # Initialize models
    embedding_model, reg_model, device = aiupred_lib.init_models(analysis_type)

    results = {}

    for header, sequence in sequences.items():
        try:
            if analysis_type == "binding":
                scores = aiupred_lib.predict_binding(sequence, embedding_model, reg_model, device)
            else:
                scores = aiupred_lib.predict_disorder(sequence, embedding_model, reg_model, device)

            scores_list = [float(s) for s in scores]
            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0

            results[header] = {
                "sequence": sequence,
                "scores": scores_list,
                "average_score": avg_score
            }

        except Exception as e:
            results[header] = {"error": str(e)}

    with open(output_path, "w") as jf:
        json.dump(results, jf, indent=4)
    print(f"[INFO] AIUPred results saved to {output_path}")

    return results


# ===== Example usage =====
if __name__ == "__main__":
    fasta_file = r"C:\Users\mteli\test.fasta"

    # ---- Run TANGO ----
    tango_results_json = r"C:\Users\mteli\tango_results.json"
    tango_exe = r"C:\Users\mteli\Downloads\tango2_3_1.winXP_\Tango.exe"
    tango_work_dir = r"C:\Users\mteli\Downloads\tango2_3_1.winXP_"

    tango_results = run_tango_from_fasta(fasta_file, tango_exe, tango_work_dir, tango_results_json)
    for seq_id, data in tango_results.items():
        print(f"{seq_id}: TANGO avg aggregation = {data['avg_agg']}")

    # ---- Run AIUPred ----
    aiupred_results_json = r"C:\Users\mteli\aiupred_results.json"
    aiupred_results = run_aiupred_local(fasta_file, aiupred_results_json, analysis_type="disorder")
    for seq_id, data in aiupred_results.items():
        print(f"{seq_id}: AIUPred avg score = {data['average_score']}")


#Add RMSD stuff and more