# CREATED BY MANAS TELIKEPALLI AND MANASHWIN NELLURI

import json
from pathlib import Path

def reward_calculator(
    mutant_json_path,
    wt_json_path,
    func_weights=None,
    agg_weights=None,
    alpha=0.5,
    functionality_cutoff=0.3,
    output_json=True
):
    # === Default Weights ===
    DEFAULT_FUNC_WEIGHTS = {"docking": 0.4, "gibbs": 0.4, "lipids": 0.2}
    DEFAULT_AGG_WEIGHTS = {
        "fibrils": 0.2, "longest_len": 0.1, "mass_frac": 0.2, "avg_len": 0.1,
        "nucleation": 0.1, "secondary": 0.1, "fragmentation": 0.1, "elongation": 0.1
    }

    # === Load auxiliary data ===
    # Fix path to work from mainTrainingLoop.py context
    DATA_DIR = Path(__file__).parent / "data"
    
    # Load individual property files instead of non-existent per_residue_properties.json
    with open(DATA_DIR / "charge.json") as f:
        CHARGE_DATA = json.load(f)
    with open(DATA_DIR / "disorder_propensity.json") as f:
        DISORDER_DATA = json.load(f)
    with open(DATA_DIR / "flexibility.json") as f:
        FLEXIBILITY_DATA = json.load(f)
    with open(DATA_DIR / "hydrophobicity.json") as f:
        HYDROPHOBICITY_DATA = json.load(f)
    with open(DATA_DIR / "secondary_structure_propensity.json") as f:
        SS_PROPENSITY_DATA = json.load(f)
    with open(DATA_DIR / "solvent_accessibility_baseline.json") as f:
        SOLVENT_DATA = json.load(f)
    with open(DATA_DIR / "volume.json") as f:
        VOLUME_DATA = json.load(f)
    
    with open(DATA_DIR / "gamma_aligned.json") as f:
        GAMMA_ALIGNED = json.load(f)
    with open(DATA_DIR / "known_pathogenic_mutations.json") as f:
        PATHOGENIC = json.load(f)

    # === Load raw input data ===
    with open(mutant_json_path) as f:
        mutant_data = json.load(f)
    with open(wt_json_path) as f:
        wt_data = json.load(f)

    func_weights = func_weights or DEFAULT_FUNC_WEIGHTS
    agg_weights = agg_weights or DEFAULT_AGG_WEIGHTS

    # === NEW: Per-residue property-based reward calculation ===
    def calculate_property_based_reward(mutations, wt_sequence):
        """
        Calculate reward based on how mutations affect per-residue properties.
        Returns a score where higher is better (more favorable properties).
        """
        if not mutations:
            return 0.0
        
        total_property_score = 0.0
        mutation_count = len(mutations)
        
        # Load WT sequence if not provided
        if not wt_sequence:
            try:
                with open(DATA_DIR / "wt_a_syn.fasta") as f:
                    lines = f.readlines()
                    wt_sequence = ''.join([l.strip() for l in lines if not l.startswith(">")])
            except:
                wt_sequence = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
        
        for mutation in mutations:
            if len(mutation) >= 3:
                mutation_type, pos, new_aa = mutation[0], mutation[1], mutation[2]
                pos_str = str(pos)
                
                if mutation_type == 'point' and 0 < pos <= len(wt_sequence):
                    old_aa = wt_sequence[pos - 1]
                    
                    # Calculate property changes
                    property_changes = {}
                    
                    # PRIORITIZED: Reduced property weights to prioritize functionality/aggregation
                    # Charge change (favor neutral or slight negative for membrane binding)
                    old_charge = CHARGE_DATA["residues"].get(old_aa, 0)
                    new_charge = CHARGE_DATA["residues"].get(new_aa, 0)
                    charge_change = abs(new_charge - old_charge)
                    # Penalize large charge changes (reduced weight)
                    property_changes["charge"] = -charge_change * 0.02
                    
                    # Hydrophobicity change (favor moderate hydrophobicity for membrane binding)
                    old_hydro = HYDROPHOBICITY_DATA["residues"].get(old_aa, 0)
                    new_hydro = HYDROPHOBICITY_DATA["residues"].get(new_aa, 0)
                    # Favor hydrophobicity around 0-2 (moderate)
                    optimal_hydro = 1.0
                    hydro_penalty = abs(new_hydro - optimal_hydro) - abs(old_hydro - optimal_hydro)
                    property_changes["hydrophobicity"] = -hydro_penalty * 0.01
                    
                    # Disorder propensity (favor moderate disorder for IDP behavior)
                    old_disorder = DISORDER_DATA["residues"].get(old_aa, 0)
                    new_disorder = DISORDER_DATA["residues"].get(new_aa, 0)
                    # Favor disorder around 0.2-0.4
                    optimal_disorder = 0.3
                    disorder_penalty = abs(new_disorder - optimal_disorder) - abs(old_disorder - optimal_disorder)
                    property_changes["disorder"] = -disorder_penalty * 0.02
                    
                    # Volume change (penalize large volume changes)
                    old_volume = VOLUME_DATA["residues"].get(old_aa, 140)
                    new_volume = VOLUME_DATA["residues"].get(new_aa, 140)
                    volume_change = abs(new_volume - old_volume) / 100.0  # Normalize
                    property_changes["volume"] = -volume_change * 0.01
                    
                    # Flexibility (favor moderate flexibility)
                    old_flex = FLEXIBILITY_DATA["residues"].get(old_aa, 0.43)
                    new_flex = FLEXIBILITY_DATA["residues"].get(new_aa, 0.43)
                    optimal_flex = 0.45
                    flex_penalty = abs(new_flex - optimal_flex) - abs(old_flex - optimal_flex)
                    property_changes["flexibility"] = -flex_penalty * 0.02
                    
                    # Secondary structure propensity (favor coil propensity for IDP)
                    old_ss = SS_PROPENSITY_DATA["residues"].get(old_aa, {"coil": 1.0})
                    new_ss = SS_PROPENSITY_DATA["residues"].get(new_aa, {"coil": 1.0})
                    old_coil = old_ss.get("coil", 1.0)
                    new_coil = new_ss.get("coil", 1.0)
                    coil_bonus = (new_coil - old_coil) * 0.05  # Reduced weight
                    property_changes["secondary_structure"] = coil_bonus
                    
                    # Solvent accessibility (favor moderate accessibility)
                    old_solvent = SOLVENT_DATA["residues"].get(old_aa, 0.45)
                    new_solvent = SOLVENT_DATA["residues"].get(new_aa, 0.45)
                    optimal_solvent = 0.5
                    solvent_penalty = abs(new_solvent - optimal_solvent) - abs(old_solvent - optimal_solvent)
                    property_changes["solvent"] = -solvent_penalty * 0.02
                    
                    # Sum up property changes
                    mutation_score = sum(property_changes.values())
                    total_property_score += mutation_score
        
        # Normalize by mutation count and return
        return total_property_score / max(mutation_count, 1)

    # === Functionality Score (relative to WT) ===
    def get_functionality_score(data, wt):
        # Docking & Gibbs are "more negative = better", so invert first
        docking_norm = (-data["docking_score"]) / (-wt["docking_score"])
        gibbs_norm  = (-data["gibbs_free_energy"]) / (-wt["gibbs_free_energy"])
        lipids_norm = data["avg_lipid_contacts"] / wt["avg_lipid_contacts"]

        func_score = (
            func_weights["docking"] * docking_norm +
            func_weights["gibbs"] * gibbs_norm +
            func_weights["lipids"] * lipids_norm
        )

        deltas = {
            "docking": docking_norm - 1.0,
            "gibbs": gibbs_norm - 1.0,
            "lipids": lipids_norm - 1.0
        }

        return round(func_score, 3), {
            "norms": {
                "docking_norm": docking_norm,
                "gibbs_norm": gibbs_norm,
                "lipids_norm": lipids_norm
            },
            "deltas": deltas
        }

    # === Aggregation Score (relative to WT, lower is better) ===
    def get_aggregation_score(data, wt):
        agg_raw = {
            "fibrils": data["fibrils"],
            "longest_len": data["longest_fibril"],
            "mass_frac": data["mass_frac"],
            "avg_len": data["avg_fibril_length"],
            "nucleation": data["nucleation"],
            "secondary": data["secondary"],
            "fragmentation": data["fragmentation"],
            "elongation": data["elongation"]
        }
        wt_raw = {
            "fibrils": wt["fibrils"],
            "longest_len": wt["longest_fibril"],
            "mass_frac": wt["mass_frac"],
            "avg_len": wt["avg_fibril_length"],
            "nucleation": wt["nucleation"],
            "secondary": wt["secondary"],
            "fragmentation": wt["fragmentation"],
            "elongation": wt["elongation"]
        }

        agg_norms = {
            k: (agg_raw[k] / wt_raw[k]) if wt_raw[k] != 0 else 1.0
            for k in agg_raw
        }
        agg_score = sum(agg_weights[k] * agg_norms[k] for k in agg_weights)

        deltas = {
            k: 1.0 - agg_norms[k]  # positive delta means less aggregation
            for k in agg_norms
        }

        return round(agg_score, 3), {"norms": agg_norms, "deltas": deltas}

    # === Calculate Scores ===
    mutant_func_score, func_components = get_functionality_score(mutant_data, wt_data)
    mutant_agg_score, agg_components = get_aggregation_score(mutant_data, wt_data)
    
    # === PRIORITIZED: Calculate property-based reward (reduced weight) ===
    # Property rewards are calculated but given minimal weight (15%) compared to 
    # functionality/aggregation scores (85%) to prioritize simulation-based outcomes
    mutations = mutant_data.get("mutations", [])
    property_reward = calculate_property_based_reward(mutations, None)

    # WT reference scores are fixed
    wt_scores = {
        "functionality_score": 1.0,
        "aggregation_score": 1.0,
        "property_score": 0.0  # NEW
    }
    mutant_scores = {
        "functionality_score": mutant_func_score,
        "aggregation_score": mutant_agg_score,
        "property_score": property_reward,  # NEW
        "functionality_components": func_components,
        "aggregation_components": agg_components
    }

    # === Deltas ===
    func_delta = round(mutant_func_score - 1.0, 3)        # want >= 0
    agg_delta = round(1.0 - mutant_agg_score, 3)          # want > 0
    property_delta = round(property_reward - 0.0, 3)      # NEW: want > 0

    # === Human-readable interpretation ===
    def interpret(delta):
        if delta > 0.1: return "strongly improved"
        elif delta > 0.01: return "slightly improved"
        elif delta < -0.1: return "strongly reduced"
        elif delta < -0.01: return "slightly reduced"
        else: return "no change"

    interpretation = {
        "functionality": interpret(func_delta),
        "aggregation": interpret(agg_delta),
        "properties": interpret(property_delta)  # NEW
    }

    # === Penalize known pathogenic mutations ===
    pathogenic_penalty = 0.0
    gamma_bonus = 0.0

    for pos, new_aa in mutations:
        pos_str = str(pos)
        if pos_str in PATHOGENIC and new_aa in PATHOGENIC[pos_str]:
            pathogenic_penalty += 0.25
        if pos_str in GAMMA_ALIGNED and new_aa == GAMMA_ALIGNED[pos_str]:
            gamma_bonus += 0.1

    # === Enhanced Combined Reward with Property Component ===
    if mutant_func_score < functionality_cutoff:
        combined_score = -1.0
    else:
        # PRIORITIZE simulation results over property-based rewards
        # Weight: 85% simulation results, 15% property-based
        simulation_component = alpha * func_delta + (1 - alpha) * agg_delta
        property_component = property_delta * 0.15  # Reduced property contribution
        
        combined_score = simulation_component * 0.85 + property_component
        combined_score += gamma_bonus
        combined_score -= pathogenic_penalty
        combined_score = round(combined_score, 3)

    # === Final Result ===
    result = {
        "mutant_scores": mutant_scores,
        "wt_scores": wt_scores,
        "delta": {
            "functionality": func_delta,
            "aggregation": agg_delta,
            "properties": property_delta,  # NEW
            "per_metric": {
                "functionality": func_components["deltas"],
                "aggregation": agg_components["deltas"]
            }
        },
        "interpretation": interpretation,
        "combined_reward": combined_score,
        "mutations": mutations,
        "gamma_bonus": gamma_bonus,
        "pathogenic_penalty": pathogenic_penalty,
        "property_reward": property_reward,  # NEW
        "parameters": {
            "func_weights": func_weights,
            "agg_weights": agg_weights,
            "alpha": alpha,
            "functionality_cutoff": functionality_cutoff
        },
        "raw_json_paths": {
            "mutant": str(mutant_json_path),
            "wt": str(wt_json_path)
        }
    }

    # === Optional Save ===
    if output_json:
        output_name = Path(mutant_json_path).with_name(Path(mutant_json_path).stem + "_reward.json")
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=2)

    return result


# === Command-line Execution ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python reward_calculator.py mutant.json wt.json")
        sys.exit(1)

    reward_calculator(sys.argv[1], sys.argv[2])