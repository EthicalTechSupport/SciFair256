# CREATED BY MANAS TELIKEPALLI

import numpy as np
import json
from pathlib import Path

def mutation_filtering_engine(
    msa_entropy_file,
    ss_regions_file,
    gamma_aligned_file, 
    entropy_threshold,
    allowed_ss={"C", "H"},
    output_json=None,
    mutation_tiers_file=None,
    known_pathogenic_file=None
):
    # === Check Input Files Exist ===
    for file in [msa_entropy_file, ss_regions_file, gamma_aligned_file, mutation_tiers_file, known_pathogenic_file]:
        if file and not Path(file).exists():
            raise FileNotFoundError(f"❌ File not found: {file}")

    # === Load entropy ===
    entropy = np.load(msa_entropy_file)
    n_residues = len(entropy)

    # === Load secondary structure map ===
    with open(ss_regions_file, 'r') as f:
        ss_map = json.load(f)

    # === Load gamma alignment ===
    with open(gamma_aligned_file, 'r') as f:
        gamma_map = json.load(f)
        gamma_aligned_positions = {int(k) for k in gamma_map if gamma_map[k]}

    # === Load mutation tiers ===
    tier_map = {}
    if mutation_tiers_file:
        with open(mutation_tiers_file, 'r') as f:
            ranges = json.load(f)
            for k, v in ranges.items():
                start, end = map(int, k.split("-"))
                for i in range(start, end + 1):
                    tier_map[i] = v

    # === Load known pathogenic mutations ===
    known_pathogenic = {}
    if known_pathogenic_file:
        with open(known_pathogenic_file, 'r') as f:
            known_pathogenic = json.load(f)
            known_pathogenic = {int(k): set(v) for k, v in known_pathogenic.items()}

    # === Lock functional membrane-binding motifs from the publication ===
    # Motifs: M1-S9 and five imperfect KTKEGV repeats (residues 10-15, 21-26, 32-37, 43-48, 58-63)
    MEMBRANE_MOTIF_RANGES = [
        (1, 9),
        (10, 15),
        (21, 26),
        (32, 37),
        (43, 48),
        (58, 63),
        (120, 140)
    ]
    LOCKED_MEMBRANE_MOTIF_POSITIONS = {pos for a, b in MEMBRANE_MOTIF_RANGES for pos in range(a, b + 1)}

    # === Determine entropy cutoff ===
    if isinstance(entropy_threshold, str) and entropy_threshold.endswith("%"):
        percentile = float(entropy_threshold.strip('%'))
        cutoff = np.percentile(entropy, percentile)
    else:
        cutoff = float(entropy_threshold)

        # === Filtering logic ===
    results = {}
    for i in range(1, 141):  # residue index 1 to 140
        res_entropy = entropy[i - 1]
        ss = ss_map.get(str(i), "C")
        is_aligned = i in gamma_aligned_positions
        tier = tier_map.get(i, "available")

        # 1) Publication motifs: always locked
        if i in LOCKED_MEMBRANE_MOTIF_POSITIONS:
            status = "locked"
            rationale = "Locked: membrane-binding functional motif (publication)"
            score = 0

        # 2) Existing rules (kept as-is, in the same order)
        elif tier == "lock":
            status = "locked"
            rationale = "Locked by tier map"
            score = 0
        elif res_entropy < cutoff:
            status = "locked"
            rationale = f"Low entropy ({res_entropy:.3f} < {cutoff:.3f})"
            score = 1
        elif ss not in allowed_ss:
            status = "locked"
            rationale = f"Disallowed secondary structure: {ss}"
            score = 1
        else:
            status = "available"
            rationale = "Passes all filters"
            score = 3
            if tier == "target":
                status = "target"
                rationale += "; prioritized by tier"
                score = 5
            elif tier == "soft":
                rationale += "; allowed with caution"
                score = 4

        # 3) Pathogenic override (keep last so it can override anything)
        if i in known_pathogenic:
            status = "locked"
            rationale = "Known pathogenic mutation(s)"
            score = 0

        results[i] = {
            "status": status,
            "entropy": res_entropy,
            "ss": ss,
            "aligned": is_aligned,
            "tier": tier,
            "score": score,
            "rationale": rationale
        }


    # === Save results if requested ===
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved mutation filtering results to: {output_path}")

    return results

# === Utility function for standalone use ===
def run_mutation_filtering_from_data_dir(
    data_dir: str = "data",
    entropy_threshold: float = 0.5,
    allowed_ss: set = {"C", "H"},
    output_json: str = "filtered_mutations.json"
):
    """
    Convenience function to run mutation filtering using files from the data directory.
    """
    data_path = Path(data_dir)
    
    return mutation_filtering_engine(
        msa_entropy_file=str(data_path / "msa_entropy.npy"),
        ss_regions_file=str(data_path / "ss_regions.json"),
        gamma_aligned_file=str(data_path / "gamma_aligned.json"),
        entropy_threshold=entropy_threshold,
        allowed_ss=allowed_ss,
        output_json=output_json,
        mutation_tiers_file=str(data_path / "mutation_tiers.json"),
        known_pathogenic_file=str(data_path / "known_pathogenic_mutations.json")
    )

if __name__ == "__main__":
    # Example usage
    results = run_mutation_filtering_from_data_dir()
    print(f"Filtered {len(results)} mutations")