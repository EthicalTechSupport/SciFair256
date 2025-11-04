# CREATED BY MANAS TELIKEPALLI
def create_mutation_agent(
    mutable_positions_path: str,
    reference_sequence_fasta: str,
    simulation_result_json_path: str | None = None,  # Made optional
    total_timesteps: int = 100000,
    func_weight: float = 1.0,
    agg_weight: float = 1.0,
    target_func_threshold: float = 0.9,
    target_agg_threshold: float = 0.2
):
    from pathlib import Path
    import json
    import numpy as np
    from typing import Tuple, Dict, List
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    import logging
    import importlib.util
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Fix path handling to work from mainTrainingLoop.py context
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    blosum_aa_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                       'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                       'T', 'W', 'Y', 'V']

    def load_sequence(path):
        with open(path, "r") as f:
            lines = f.readlines()
        return ''.join([l.strip() for l in lines if not l.startswith(">")])

    def load_jsons():
        wt_seq = load_sequence(reference_sequence_fasta)

        with open(mutable_positions_path) as f:
            all_positions = json.load(f)

        mutable_positions = {
            pos: {
                "wt": wt_seq[int(pos) - 1],
                "status": v.get("status", "")
            }
            for pos, v in all_positions.items()
            if v.get("status") in {"available", "target"}
        }

        # Load supporting resources with safe fallbacks
        def safe_json(p, default={}):
            try:
                return json.load(open(p))
            except Exception:
                return default

        def safe_npy(p, default=None):
            try:
                return np.load(p)
            except Exception:
                return default

        return {
            "wt_seq": wt_seq,
            "mutable": mutable_positions,
            "forbidden": safe_json(DATA_DIR / "known_pathogenic_mutations.json", {}),
            "gamma": safe_json(DATA_DIR / "gamma_aligned.json", {}),
            "blosum": safe_npy(DATA_DIR / "blosum62.npy", np.zeros((20, 20))),
            "charge": safe_json(DATA_DIR / "charge.json", {}),
            "disorder": safe_json(DATA_DIR / "disorder_propensity.json", {}),
            "flexibility": safe_json(DATA_DIR / "flexibility.json", {}),
            "hydrophobicity": safe_json(DATA_DIR / "hydrophobicity.json", {}),
            "msa_entropy": safe_npy(DATA_DIR / "msa_entropy.npy", np.zeros(len(wt_seq))),
            "secstruct": safe_json(DATA_DIR / "secondary_structure_propensity.json", {}),
            "solvent": safe_json(DATA_DIR / "solvent_accessibility_baseline.json", {}),
            "ss_regions": safe_json(DATA_DIR / "ss_regions.json", {}),
            "volume": safe_json(DATA_DIR / "volume.json", {}),
        }

    data = load_jsons()

    sorted_mutables = sorted(
        data["mutable"].items(),
        key=lambda kv: 0 if kv[1]["status"] == "target" else 1
    )

    action_map = []

    # Build action map with proper gamma scoring
    for pos_str, meta in sorted_mutables:
        wt_aa = meta["wt"]
        pos = int(pos_str)

        # Point mutations with improved gamma bonus
        for aa in amino_acids:
            if aa != wt_aa and not (pos_str in data["forbidden"] and aa in data["forbidden"][pos_str]):
                gamma_bonus = 2.0 if data["gamma"].get(pos_str) == aa else 0.0
                # Add conservation bonus for conservative mutations
                try:
                    blosum_score = data["blosum"][blosum_aa_order.index(wt_aa)][blosum_aa_order.index(aa)]
                    conservation_bonus = max(0, blosum_score / 5.0)  # Normalize BLOSUM scores
                except (ValueError, IndexError):
                    conservation_bonus = 0.0

                # PRIORITIZED: Functionality/aggregation over biophysical properties
                # Primary: Gamma alignment (strongest signal)
                # Secondary: Conservation bonus (BLOSUM-based)
                # Tertiary: Property optimization (reduced weight)
                
                property_bonus = 0.0
                try:
                    # Charge optimization (favor neutral or slight negative for membrane binding)
                    old_charge = data["charge"].get(pos_str, 0)
                    new_charge = data["charge"]["residues"].get(aa, 0)
                    charge_optimal = 0.0  # Neutral is optimal
                    charge_penalty = abs(new_charge - charge_optimal) - abs(old_charge - charge_optimal)
                    property_bonus += -charge_penalty * 0.02  # Reduced weight
                    
                    # Hydrophobicity optimization (favor moderate hydrophobicity)
                    old_hydro = data["hydrophobicity"].get(pos_str, 0)
                    new_hydro = data["hydrophobicity"]["residues"].get(aa, 0)
                    hydro_optimal = 1.0  # Moderate hydrophobicity
                    hydro_penalty = abs(new_hydro - hydro_optimal) - abs(old_hydro - hydro_optimal)
                    property_bonus += -hydro_penalty * 0.01  # Reduced weight
                    
                    # Disorder propensity (favor moderate disorder for IDP)
                    old_disorder = data["disorder"].get(pos_str, 0)
                    new_disorder = data["disorder"]["residues"].get(aa, 0)
                    disorder_optimal = 0.3  # Moderate disorder
                    disorder_penalty = abs(new_disorder - disorder_optimal) - abs(old_disorder - disorder_optimal)
                    property_bonus += -disorder_penalty * 0.02  # Reduced weight
                    
                    # Volume change penalty (favor smaller changes)
                    old_volume = data["volume"].get(pos_str, 140)
                    new_volume = data["volume"]["residues"].get(aa, 140)
                    volume_change = abs(new_volume - old_volume) / 100.0
                    property_bonus += -volume_change * 0.01  # Reduced weight
                    
                    # Flexibility optimization
                    old_flex = data["flexibility"].get(pos_str, 0.43)
                    new_flex = data["flexibility"]["residues"].get(aa, 0.43)
                    flex_optimal = 0.45
                    flex_penalty = abs(new_flex - flex_optimal) - abs(old_flex - flex_optimal)
                    property_bonus += -flex_penalty * 0.02  # Reduced weight
                    
                    # Secondary structure (favor coil propensity)
                    old_ss = data["secstruct"].get(pos_str, {"coil": 1.0})
                    new_ss = data["secstruct"]["residues"].get(aa, {"coil": 1.0})
                    if isinstance(old_ss, dict) and isinstance(new_ss, dict):
                        old_coil = old_ss.get("coil", 1.0)
                        new_coil = new_ss.get("coil", 1.0)
                        coil_bonus = (new_coil - old_coil) * 0.05  # Reduced weight
                        property_bonus += coil_bonus
                    
                    # Solvent accessibility optimization
                    old_solvent = data["solvent"].get(pos_str, 0.45)
                    new_solvent = data["solvent"]["residues"].get(aa, 0.45)
                    solvent_optimal = 0.5
                    solvent_penalty = abs(new_solvent - solvent_optimal) - abs(old_solvent - solvent_optimal)
                    property_bonus += -solvent_penalty * 0.02  # Reduced weight
                    
                except Exception:
                    # Fallback if property calculation fails
                    property_bonus = 0.0

                # PRIORITIZED BONUS CALCULATION:
                # 1. Gamma alignment (highest priority - 60% weight)
                # 2. Conservation bonus (secondary - 30% weight) 
                # 3. Property optimization (tertiary - 10% weight)
                total_bonus = (gamma_bonus * 0.6) + (conservation_bonus * 0.3) + (property_bonus * 0.1)
                action_map.append(('point', pos, aa, total_bonus))

        # Insertions (more conservative approach)
        for aa in amino_acids:
            # Small penalty for insertions as they're more disruptive
            action_map.append(('insert', pos, aa, -0.5))

        # Deletions (larger penalty as they're very disruptive)
        action_map.append(('delete', pos, None, -1.0))

    index_to_action = {
        i: (action_type, pos, aa, bonus)
        for i, (action_type, pos, aa, bonus) in enumerate(action_map)
    }

    class MultiObjectiveReward:
        """Handles multi-objective reward calculation (legacy wrapper)"""

        def __init__(self, func_weight: float, agg_weight: float,
                     target_func: float, target_agg: float):
            self.func_weight = func_weight
            self.agg_weight = agg_weight
            self.target_func = target_func
            self.target_agg = target_agg

        def calculate_reward(self, func_score: float, agg_score: float,
                             step_penalty: float = 0.0, action_bonus: float = 0.0) -> Dict:
            """Calculate multi-objective reward with different strategies (fallback)"""

            # Strategy 1: Weighted sum with normalization
            func_normalized = min(func_score / self.target_func, 2.0) if self.target_func > 0 else func_score
            agg_normalized = max(0, (self.target_agg - agg_score) / self.target_agg) if self.target_agg > 0 else 0.0

            weighted_reward = (self.func_weight * func_normalized +
                               self.agg_weight * agg_normalized) / (self.func_weight + self.agg_weight + 1e-12)

            # Strategy 2: Pareto dominance bonus
            pareto_bonus = 0.0
            if func_score >= self.target_func and agg_score <= self.target_agg:
                pareto_bonus = 2.0  # Strong bonus for meeting both objectives
            elif func_score >= self.target_func * 0.8 and agg_score <= self.target_agg * 1.5:
                pareto_bonus = 1.0  # Moderate bonus for close to targets

            # Strategy 3: Constraint-based penalty
            constraint_penalty = 0.0
            if func_score < 0.5:  # Severe functionality loss
                constraint_penalty = -2.0
            if agg_score > 0.8:  # Severe aggregation
                constraint_penalty += -2.0

            total_reward = weighted_reward + pareto_bonus + constraint_penalty + action_bonus - step_penalty

            return {
                'total_reward': total_reward,
                'weighted_reward': weighted_reward,
                'pareto_bonus': pareto_bonus,
                'constraint_penalty': constraint_penalty,
                'func_score': func_score,
                'agg_score': agg_score,
                'func_normalized': func_normalized,
                'agg_normalized': agg_normalized
            }

    # Try to load reward_calculator if present (optional)
    reward_calculator = None
    try:
        # prefer local module if importable
        from rewardCalculation import reward_calculator as _rc  # type: ignore
        reward_calculator = _rc
    except Exception:
        # try typical upload location
        candidate_paths = [
            Path(__file__).resolve().parent / "rewardCalculation.py",
            Path("/mnt/data/rewardCalculation.py"),
            BASE_DIR / "rewardCalculation.py"
        ]
        for p in candidate_paths:
            try:
                if p.exists():
                    spec = importlib.util.spec_from_file_location("rewardCalculation_loaded", str(p))
                    if spec is not None:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules["rewardCalculation_loaded"] = module
                        spec.loader.exec_module(module)  # type: ignore
                        reward_calculator = getattr(module, "reward_calculator")
                    break
            except Exception:
                continue

    class AdaptiveMutationEnv(gym.Env):
        """Improved environment with adaptive rewards and better state representation"""

        def __init__(self, sim_json_path: str | None = None):
            super().__init__()
            self.sim_result_path = Path(sim_json_path) if sim_json_path is not None else None
            self.original_seq = list(data["wt_seq"])
            self.current_seq = self.original_seq.copy()
            self.mutations = []
            self.max_mutations = 8  # Increased for more flexibility
            self.step_count = 0
            self.seq_len = len(self.original_seq)
            self.num_features = 26  # Enhanced feature space with property integration

            # Improved observation space
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0,
                shape=(self.seq_len * 26 + 20,),  # 26 features per residue + global features
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(len(action_map))

            # Initialize reward calculator object (fallback wrapper)
            self.reward_calc = MultiObjectiveReward(
                func_weight, agg_weight, target_func_threshold, target_agg_threshold
            )

            # Track mutation history for better exploration
            self.mutation_history = []
            self.best_reward = float('-inf')
            self.episode_rewards = []

        def extract_features(self, seq: List[str]) -> np.ndarray:
            """Enhanced feature extraction with better property integration"""
            # Sequence-level features
            seq_features = []
            for i, aa in enumerate(seq[:self.seq_len]):
                aa_str = aa if aa in amino_acids else "X"
                pos_str = str(i + 1)
                wt = self.original_seq[i] if i < len(self.original_seq) else aa_str

                # Enhanced features
                try:
                    blosum_score = float(data["blosum"][blosum_aa_order.index(wt)][blosum_aa_order.index(aa_str)]) / 10
                except (ValueError, IndexError):
                    blosum_score = 0.0

                entropy = float(data["msa_entropy"][i]) if i < len(data["msa_entropy"]) else 0.0

                # Mutation indicator
                is_mutated = 1.0 if aa_str != wt else 0.0

                # NEW: Enhanced property features with more detailed information
                # Charge features
                charge_val = data["charge"]["residues"].get(aa_str, 0)
                charge_deviation = abs(charge_val - 0.0)  # Deviation from neutral
                
                # Hydrophobicity features
                hydro_val = data["hydrophobicity"]["residues"].get(aa_str, 0)
                hydro_optimality = 1.0 - abs(hydro_val - 1.0) / 5.0  # Closer to optimal = higher
                
                # Disorder features
                disorder_val = data["disorder"]["residues"].get(aa_str, 0)
                disorder_optimality = 1.0 - abs(disorder_val - 0.3) / 1.0  # Closer to optimal = higher
                
                # Volume features
                volume_val = data["volume"]["residues"].get(aa_str, 140)
                volume_normalized = (volume_val - 60) / (230 - 60)  # Normalize to 0-1
                
                # Flexibility features
                flex_val = data["flexibility"]["residues"].get(aa_str, 0.43)
                flex_optimality = 1.0 - abs(flex_val - 0.45) / 0.5  # Closer to optimal = higher
                
                # Secondary structure features
                ss_data = data["secstruct"]["residues"].get(aa_str, {"helix": 1.0, "sheet": 1.0, "coil": 1.0})
                if isinstance(ss_data, dict):
                    coil_propensity = ss_data.get("coil", 1.0)
                    helix_propensity = ss_data.get("helix", 1.0)
                    sheet_propensity = ss_data.get("sheet", 1.0)
                else:
                    coil_propensity = 1.0
                    helix_propensity = 1.0
                    sheet_propensity = 1.0
                
                # Solvent accessibility features
                solvent_val = data["solvent"]["residues"].get(aa_str, 0.45)
                solvent_optimality = 1.0 - abs(solvent_val - 0.5) / 0.5  # Closer to optimal = higher

                seq_features.extend([
                    aa_to_idx.get(aa_str, 0) / 20,
                    blosum_score,
                    data["charge"].get(pos_str, 0),
                    data["disorder"].get(pos_str, 0),
                    data["flexibility"].get(pos_str, 0),
                    data["hydrophobicity"].get(pos_str, 0),
                    entropy,
                    data["secstruct"].get(pos_str, 0),
                    data["solvent"].get(pos_str, 0),
                    data["volume"].get(pos_str, 0),
                    is_mutated,
                    len([m for m in self.mutations if m[1] == i + 1]) / self.max_mutations,  # Mutation density
                    # NEW: Enhanced property features
                    charge_val,
                    charge_deviation,
                    hydro_val,
                    hydro_optimality,
                    disorder_val,
                    disorder_optimality,
                    volume_normalized,
                    flex_val,
                    flex_optimality,
                    coil_propensity,
                    helix_propensity,
                    sheet_propensity,
                    solvent_val,
                    solvent_optimality
                ])

            # Pad to fixed length (increased for new features)
            while len(seq_features) < self.seq_len * 26:  # Increased from 12 to 26 features
                seq_features.append(0.0)

            # Global features
            global_features = [
                len(self.mutations) / self.max_mutations,  # Progress
                len(seq) / self.seq_len,  # Length change
                sum(1 for m in self.mutations if m[0] == 'point') / max(len(self.mutations), 1),  # Point mut ratio
                sum(1 for m in self.mutations if m[0] == 'insert') / max(len(self.mutations), 1),  # Insert ratio
                sum(1 for m in self.mutations if m[0] == 'delete') / max(len(self.mutations), 1),  # Delete ratio
                self.step_count / self.max_mutations,  # Time progress
                self.best_reward / 10.0 if self.best_reward > float('-inf') else 0.0,  # Best reward so far
                len(set(m[1] for m in self.mutations)) / max(len(self.mutations), 1),  # Position diversity
                # NEW: Global property statistics
                np.mean([data["charge"]["residues"].get(aa, 0) for aa in seq[:self.seq_len]]),  # Avg charge
                np.std([data["charge"]["residues"].get(aa, 0) for aa in seq[:self.seq_len]]),   # Charge variance
                np.mean([data["hydrophobicity"]["residues"].get(aa, 0) for aa in seq[:self.seq_len]]),  # Avg hydrophobicity
                np.mean([data["disorder"]["residues"].get(aa, 0) for aa in seq[:self.seq_len]]),  # Avg disorder
                np.mean([data["volume"]["residues"].get(aa, 140) for aa in seq[:self.seq_len]]) / 200,  # Avg volume (normalized)
                np.mean([data["flexibility"]["residues"].get(aa, 0.43) for aa in seq[:self.seq_len]]),  # Avg flexibility
                np.mean([data["solvent"]["residues"].get(aa, 0.45) for aa in seq[:self.seq_len]]),  # Avg solvent accessibility
            ]

            # Pad global features
            while len(global_features) < 20:
                global_features.append(0.0)

            return np.array(seq_features[:self.seq_len * 26] + global_features[:20], dtype=np.float32)

        def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self.current_seq = self.original_seq.copy()
            self.mutations = []
            self.step_count = 0
            if self.episode_rewards:
                self.best_reward = max(self.episode_rewards)
            self.episode_rewards = []
            return self.extract_features(self.current_seq), {}

        def step(self, action: int):
            action_type, pos, aa, action_bonus = index_to_action[int(action)]

            # Apply mutation
            mutation_applied = False
            if action_type == 'point' and 0 < pos <= len(self.current_seq):
                old_aa = self.current_seq[pos - 1]
                self.current_seq[pos - 1] = aa
                self.mutations.append(('point', pos, aa, old_aa))
                mutation_applied = True
            elif action_type == 'insert':
                self.current_seq.insert(min(pos - 1, len(self.current_seq)), aa)
                self.mutations.append(('insert', pos, aa, None))
                mutation_applied = True
            elif action_type == 'delete' and 0 < pos <= len(self.current_seq):
                deleted = self.current_seq.pop(pos - 1)
                self.mutations.append(('delete', pos, deleted, None))
                mutation_applied = True

            self.step_count += 1

            # Step penalty to encourage efficiency
            step_penalty = 0.1 * self.step_count / self.max_mutations

            # Small immediate reward for valid mutations
            immediate_reward = 0.1 if mutation_applied else -0.5

            done = self.step_count >= self.max_mutations

            if done:
                try:
                    reward_info = self.evaluate()
                    final_reward = reward_info.get('total_reward', reward_info.get('combined_reward', -10.0))
                    # store best reward history
                    if final_reward is not None:
                        self.episode_rewards.append(final_reward)
                        if final_reward > self.best_reward:
                            self.best_reward = final_reward
                    # Return observation, reward, terminated, truncated, info (gymnasium signature)
                    return (self.extract_features(self.current_seq),
                            float(final_reward), True, False, reward_info)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    return (self.extract_features(self.current_seq),
                            -10.0, True, False, {'error': str(e)})

            return (self.extract_features(self.current_seq),
                    immediate_reward, False, False,
                    {'step_penalty': step_penalty, 'action_bonus': action_bonus})

        def evaluate(self) -> Dict:
            """
            Enhanced evaluation with separation of objectives:
             - If the simulation/result JSON already contains rewardCalculation-style output
               (i.e. 'mutant_scores' and 'delta'), we use those fields directly.
             - Else, if reward_calculator() is available, call it with mutant & wt paths.
             - Else, fall back to legacy MultiObjectiveReward using available numeric fields.
            The environment sets:
              - 'func_score' (higher is better, 0..1)
              - 'agg_score'  (lower is better, 0..1)
            For PPO training we produce a scalar where higher == better:
              PPO_reward = (func_score + (1 - agg_score)) / 2
            The returned dict contains both the scalar in 'total_reward' and the raw objectives
            in 'objectives' and 'per_metric' for mutation-policy use.
            """
            # If the path doesn't exist or is None, fallback to poor scores
            if self.sim_result_path is None or not self.sim_result_path.exists():
                if self.sim_result_path is None:
                    logger.warning("No simulation result path provided, using fallback scores")
                else:
                    logger.warning(f"Missing simulation file: {self.sim_result_path}")
                fallback = self.reward_calc.calculate_reward(0.0, 1.0, 0.0, 0.0)
                # Build consistent return
                func_score = fallback.get('func_score', 0.0)
                agg_score = fallback.get('agg_score', 1.0)
                ppo_reward = (func_score + (1.0 - agg_score)) / 2.0
                fallback.update({
                    'functionality_score': func_score,
                    'aggregation_score': agg_score,
                    'func_score': func_score,
                    'agg_score': agg_score,
                    'objectives': (func_score, agg_score),
                    'per_metric': {'functionality': [], 'aggregation': []},
                    'total_reward': ppo_reward,
                    'mutations': self.mutations,
                    'sequence': ''.join(self.current_seq)
                })
                return fallback

            # Read the existing JSON
            try:
                if self.sim_result_path is None:
                    raise FileNotFoundError("No simulation result path provided")
                with open(self.sim_result_path) as f:
                    reward_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading simulation results: {e}")
                fallback = self.reward_calc.calculate_reward(0.0, 1.0, 0.0, 0.0)
                func_score = fallback.get('func_score', 0.0)
                agg_score = fallback.get('agg_score', 1.0)
                ppo_reward = (func_score + (1.0 - agg_score)) / 2.0
                fallback.update({
                    'functionality_score': func_score,
                    'aggregation_score': agg_score,
                    'func_score': func_score,
                    'agg_score': agg_score,
                    'objectives': (func_score, agg_score),
                    'per_metric': {'functionality': [], 'aggregation': []},
                    'total_reward': ppo_reward,
                    'mutations': self.mutations,
                    'sequence': ''.join(self.current_seq)
                })
                return fallback

            # Case A: rewardCalculation-style JSON already present
            if isinstance(reward_data, dict) and "mutant_scores" in reward_data and "delta" in reward_data:
                # Extract the two canonical objective scores
                try:
                    func_score = reward_data["mutant_scores"].get("functionality_score",
                                                                  reward_data["delta"].get("functionality", 0.0))
                except Exception:
                    func_score = reward_data.get("functionality_score", 0.0)
                try:
                    agg_score = reward_data["mutant_scores"].get("aggregation_score",
                                                                 reward_data["delta"].get("aggregation", 1.0))
                except Exception:
                    agg_score = reward_data.get("aggregation_score", 1.0)

                # Ensure bounds
                func_score = float(func_score)
                agg_score = float(agg_score)

                # PPO scalar (higher is better): combine functionality (↑) and inverse aggregation (↓)
                ppo_reward = (func_score + (1.0 - agg_score)) / 2.0

                # per-metric deltas (if present)
                per_metric = reward_data.get("delta", {}).get("per_metric", {})
                if per_metric is None:
                    per_metric = {'functionality': [], 'aggregation': []}

                info = {
                    "func_score": func_score,
                    "aggregation_score": agg_score,
                    "agg_score": agg_score,
                    "functionality_score": func_score,
                    "objectives": (func_score, agg_score),
                    "per_metric": per_metric,
                    "mutations": reward_data.get("mutations", self.mutations),
                    "sequence": ''.join(self.current_seq),
                    # keep combined_reward available but DO NOT use it for PPO learning
                    "combined_reward": reward_data.get("combined_reward", None),
                    # scalar used by PPO (higher is better)
                    "total_reward": ppo_reward,
                    # keep raw reward json for debugging
                    "raw_json": reward_data
                }
                return info

            # Case B: try to call reward_calculator function if available
            if reward_calculator is not None:
                # find a WT json path candidate; many setups store wt sim in DATA_DIR
                wt_candidates = [
                    DATA_DIR / "wt_simulation.json",
                    DATA_DIR / "wt.json",
                    Path(reference_sequence_fasta).with_suffix(".json")
                ]
                # Add parent directory if sim_result_path exists
                if self.sim_result_path is not None:
                    wt_candidates.insert(0, self.sim_result_path.parent / "wt_simulation.json")
                wt_path = None
                for p in wt_candidates:
                    try:
                        if p and p.exists():
                            wt_path = p
                            break
                    except Exception:
                        continue
                wt_path_to_pass = str(wt_path) if wt_path is not None else str(DATA_DIR / "wt_simulation.json")

                try:
                    if self.sim_result_path is None:
                        raise FileNotFoundError("No simulation result path provided")
                    rc_result = reward_calculator(str(self.sim_result_path), wt_path_to_pass, func_weights=None, agg_weights=None, output_json=False)
                    # Pull the two scores
                    func_score = rc_result.get("mutant_scores", {}).get("functionality_score", rc_result.get("functionality_score", 0.0))
                    agg_score = rc_result.get("mutant_scores", {}).get("aggregation_score", rc_result.get("aggregation_score", 1.0))
                    func_score = float(func_score)
                    agg_score = float(agg_score)
                    ppo_reward = (func_score + (1.0 - agg_score)) / 2.0
                    per_metric = rc_result.get("delta", {}).get("per_metric", {})
                    info = {
                        "func_score": func_score,
                        "aggregation_score": agg_score,
                        "agg_score": agg_score,
                        "functionality_score": func_score,
                        "objectives": (func_score, agg_score),
                        "per_metric": per_metric,
                        "mutations": rc_result.get("mutations", self.mutations),
                        "sequence": ''.join(self.current_seq),
                        "combined_reward": rc_result.get("combined_reward", None),
                        "total_reward": ppo_reward,
                        "rc_full": rc_result
                    }
                    return info
                except Exception as e:
                    logger.warning(f"reward_calculator call failed: {e}")

            # Case C: legacy fallback - try to interpret reward_data numeric fields and use MultiObjectiveReward
            try:
                func_score = reward_data.get("functionality_score", reward_data.get("func_score", 0.0))
                agg_score = reward_data.get("aggregation_score", reward_data.get("agg_score", 1.0))
                func_score = float(func_score)
                agg_score = float(agg_score)

                # Compute action_bonus if possible (sum of bonuses for applied actions)
                action_bonus = 0.0
                try:
                    # action indices present in self.mutations are tuples; we attempt to sum matching action_map bonuses
                    for m in self.mutations:
                        # m is ('point'|'insert'|'delete', pos, aa, maybe old)
                        # find matching entries in index_to_action to sum bonuses (best-effort)
                        for entry in index_to_action.values():
                            if entry[0] == m[0] and entry[1] == m[1] and (entry[2] == m[2] or m[2] is None):
                                action_bonus += float(entry[3] if entry[3] is not None else 0.0)
                                break
                except Exception:
                    action_bonus = 0.0

                fallback_reward = self.reward_calc.calculate_reward(func_score, agg_score, 0.0, action_bonus)
                # But override total_reward with PPO-compatible scalar and expose raw objectives
                ppo_reward = (func_score + (1.0 - agg_score)) / 2.0
                fallback_reward.update({
                    "functionality_score": func_score,
                    "aggregation_score": agg_score,
                    "func_score": func_score,
                    "agg_score": agg_score,
                    "objectives": (func_score, agg_score),
                    "per_metric": reward_data.get("delta", {}).get("per_metric", {}),
                    "total_reward": ppo_reward,
                    "mutations": reward_data.get("mutations", self.mutations),
                    "sequence": ''.join(self.current_seq)
                })
                return fallback_reward

            except Exception as e:
                logger.error(f"Unhandled exception in evaluate(): {e}")
                fallback = self.reward_calc.calculate_reward(0.0, 1.0, 0.0, 0.0)
                func_score = fallback.get('func_score', 0.0)
                agg_score = fallback.get('agg_score', 1.0)
                ppo_reward = (func_score + (1.0 - agg_score)) / 2.0
                fallback.update({
                    'functionality_score': func_score,
                    'aggregation_score': agg_score,
                    'func_score': func_score,
                    'agg_score': agg_score,
                    'objectives': (func_score, agg_score),
                    'per_metric': {'functionality': [], 'aggregation': []},
                    'total_reward': ppo_reward,
                    'mutations': self.mutations,
                    'sequence': ''.join(self.current_seq)
                })
                return fallback

    class RewardLoggingCallback(BaseCallback):
        """Callback to log training progress"""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.func_scores = []
            self.agg_scores = []

        def _on_step(self) -> bool:
            # Log episode completion
            # Note: stable-baselines internals can pack dones/infos differently; handle safely.
            dones = self.locals.get('dones')
            infos = self.locals.get('infos')
            if dones is None or infos is None:
                return True

            # dones may be array-like; infos is a list of dicts
            try:
                if isinstance(dones, (list, tuple, np.ndarray)) and len(dones) > 0 and dones[0]:
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    if isinstance(info, dict) and ('func_score' in info or 'functionality_score' in info):
                        func = info.get('func_score', info.get('functionality_score'))
                        agg = info.get('agg_score', info.get('aggregation_score'))
                        total = info.get('total_reward', info.get('combined_reward'))
                        if func is not None and agg is not None and total is not None:
                            self.func_scores.append(func)
                            self.agg_scores.append(agg)
                            self.episode_rewards.append(total)

                            if len(self.episode_rewards) % 100 == 0:
                                avg_reward = np.mean(self.episode_rewards[-100:])
                                avg_func = np.mean(self.func_scores[-100:])
                                avg_agg = np.mean(self.agg_scores[-100:])
                                logger.info(f"Episode {len(self.episode_rewards)}: "
                                            f"Avg Reward={avg_reward:.3f}, "
                                            f"Avg Func={avg_func:.3f}, "
                                            f"Avg Agg={avg_agg:.3f}")
            except Exception:
                # keep callback robust
                pass
            return True

    # Create environment and model
    env = DummyVecEnv([lambda: AdaptiveMutationEnv(simulation_result_json_path)])

    # Enhanced PPO configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=[512, 512, 256]  # Larger network
        )
    )

    # Train with callback
    callback = RewardLoggingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    class ImprovedMutationAgent:
        """Enhanced mutation agent with better learning and exploration"""

        def __init__(self, model, index_to_action, original_seq, sim_result_path=None):
            self.model = model
            self.index_to_action = index_to_action
            self.original_seq = list(original_seq)
            self.max_mutations = 8
            self.env = DummyVecEnv([lambda: AdaptiveMutationEnv(sim_result_path)])

            # Enhanced replay buffer with priority sampling
            self.replay_buffer = []
            self.success_threshold = 0.8
            self.pareto_archive = []  # Store Pareto optimal solutions

        def select_mutations(self, history: List | None = None, num_candidates: int = 5) -> List:
            """Generate multiple candidate mutation sets"""
            candidates = []

            for _ in range(num_candidates):
                try:
                    reset_result = self.env.reset()
                    if isinstance(reset_result, tuple):
                        obs = reset_result[0]
                    else:
                        obs = reset_result
                except Exception:
                    # Fallback if reset fails
                    try:
                        obs_shape = self.env.observation_space.shape[0] if (hasattr(self.env.observation_space, 'shape') and 
                                                                          self.env.observation_space.shape is not None and 
                                                                          len(self.env.observation_space.shape) > 0) else 100
                    except (AttributeError, TypeError, IndexError):
                        obs_shape = 100
                    obs = np.zeros((1, obs_shape))
                mutations = []
                seen_positions = set()

                current_seq = self.original_seq.copy()
                steps = 0

                while steps < self.max_mutations:
                    # Use model to predict action
                    action, _ = self.model.predict(obs, deterministic=False)
                    
                    # Handle action indexing safely
                    try:
                        if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                            action_idx = int(action[0])
                        elif action is not None:
                            action_idx = int(action)  # type: ignore
                        else:
                            action_idx = 0
                    except (ValueError, TypeError, IndexError):
                        action_idx = 0
                    
                    action_type, pos, aa, bonus = self.index_to_action[action_idx]

                    # Skip if position already mutated (unless it's beneficial)
                    if pos in seen_positions and bonus < 1.0:
                        continue

                    seen_positions.add(pos)

                    # Apply mutation to tracking sequence
                    if action_type == 'point' and 0 < pos <= len(current_seq):
                        current_seq[pos - 1] = aa
                        mutations.append(('point', pos, aa))
                    elif action_type == 'insert':
                        current_seq.insert(min(pos - 1, len(current_seq)), aa)
                        mutations.append(('insert', pos, aa))
                    elif action_type == 'delete' and 0 < pos <= len(current_seq):
                        current_seq.pop(pos - 1)
                        mutations.append(('delete', pos, None))

                    # Update observation
                    env_instance = self.env.envs[0]
                    try:
                        # Try to use extract_features method
                        obs = env_instance.extract_features(current_seq).reshape(1, -1)  # type: ignore
                    except (AttributeError, TypeError):
                        # Fallback: create a simple observation
                        try:
                            obs_shape = env_instance.observation_space.shape[0] if (hasattr(env_instance.observation_space, 'shape') and 
                                                                                  env_instance.observation_space.shape is not None and 
                                                                                  len(env_instance.observation_space.shape) > 0) else 100
                        except (AttributeError, TypeError, IndexError):
                            obs_shape = 100
                        obs = np.zeros((1, obs_shape))
                    steps += 1

                candidates.append({
                    'mutations': mutations,
                    'sequence': ''.join(current_seq),
                    'diversity_score': len(seen_positions) / len(mutations) if mutations else 0
                })

            # Return the most diverse candidate
            return max(candidates, key=lambda x: x['diversity_score'])['mutations']

        def select_mutation(self, history: List | None = None) -> List:
            """Legacy method for backward compatibility - returns first mutation from select_mutations"""
            mutations = self.select_mutations(history, num_candidates=1)
            return mutations

        def apply_mutation(self, mutation_list: List) -> str:
            """Legacy method for backward compatibility - calls apply_mutations"""
            return self.apply_mutations(mutation_list)

        def apply_mutations(self, mutation_list: List) -> str:
            """Apply mutations to sequence with proper ordering"""
            seq = self.original_seq.copy()

            # Sort mutations by position (reverse for deletions to maintain indices)
            deletions = [m for m in mutation_list if m[0] == 'delete']
            insertions = [m for m in mutation_list if m[0] == 'insert']
            points = [m for m in mutation_list if m[0] == 'point']

            # Apply point mutations first
            for mutation in points:
                _, pos, aa = mutation
                if 0 < pos <= len(seq):
                    seq[pos - 1] = aa

            # Apply deletions (reverse order to maintain indices)
            for mutation in sorted(deletions, key=lambda x: x[1], reverse=True):
                _, pos, _ = mutation
                if 0 < pos <= len(seq):
                    seq.pop(pos - 1)

            # Apply insertions (forward order)
            for mutation in sorted(insertions, key=lambda x: x[1]):
                _, pos, aa = mutation
                seq.insert(min(pos - 1, len(seq)), aa)

            return ''.join(seq)

        def update_pareto_archive(self, solution: Dict):
            """Maintain archive of Pareto optimal solutions"""
            func_score = solution["reward"].get("functionality_score", 0)
            agg_score = solution["reward"].get("aggregation_score", 1)

            # Remove dominated solutions
            self.pareto_archive = [
                sol for sol in self.pareto_archive
                if not (func_score >= sol["func_score"] and agg_score <= sol["agg_score"] and
                        (func_score > sol["func_score"] or agg_score < sol["agg_score"]))
            ]

            # Add if not dominated
            is_dominated = any(
                sol["func_score"] >= func_score and sol["agg_score"] <= agg_score and
                (sol["func_score"] > func_score or sol["agg_score"] < agg_score)
                for sol in self.pareto_archive
            )

            if not is_dominated:
                solution_copy = solution.copy()
                solution_copy["func_score"] = func_score
                solution_copy["agg_score"] = agg_score
                self.pareto_archive.append(solution_copy)

                # Limit archive size
                if len(self.pareto_archive) > 50:
                    # Keep most diverse solutions
                    self.pareto_archive.sort(key=lambda x: x["func_score"] - x["agg_score"], reverse=True)
                    self.pareto_archive = self.pareto_archive[:50]

        def learn_from_history(self, history: List[Dict], fine_tune_steps: int = 2000):
            """Enhanced learning from historical data"""
            high_quality_samples = []

            for h in history:
                func_score = h["reward"].get("functionality_score", 0)
                agg_score = h["reward"].get("aggregation_score", 1)

                # Update Pareto archive
                self.update_pareto_archive(h)

                # Define success criteria (multi-objective)
                is_high_func = func_score >= target_func_threshold * 0.8
                is_low_agg = agg_score <= target_agg_threshold * 1.2

                if is_high_func and is_low_agg:
                    # Calculate priority based on dominance
                    priority = func_score - agg_score + (1.0 if func_score >= target_func_threshold and agg_score <= target_agg_threshold else 0.0)
                    h["replay_priority"] = priority
                    high_quality_samples.append(h)

            if high_quality_samples:
                logger.info(f"Fine-tuning with {len(high_quality_samples)} high-quality samples")
                logger.info(f"Pareto archive size: {len(self.pareto_archive)}")

                # Sort by priority
                high_quality_samples.sort(key=lambda x: x["replay_priority"], reverse=True)
                self.replay_buffer.extend(high_quality_samples[:20])  # Limit buffer size

                # Fine-tune the model
                self.model.set_env(self.env)
                self.model.learn(total_timesteps=fine_tune_steps)

                # Log best solutions
                for i, sol in enumerate(self.pareto_archive[:5]):
                    logger.info(f"Pareto solution {i+1}: Func={sol['func_score']:.3f}, Agg={sol['agg_score']:.3f}")
            else:
                logger.info("No high-quality samples found for fine-tuning")

        def get_best_solutions(self, n: int = 5) -> List[Dict]:
            """Return best solutions from Pareto archive"""
            return sorted(self.pareto_archive,
                          key=lambda x: x["func_score"] - x["agg_score"],
                          reverse=True)[:n]

        def update_simulation_path(self, new_sim_path: str):
            """Update the simulation result path for subsequent episodes"""
            self.env = DummyVecEnv([lambda: AdaptiveMutationEnv(new_sim_path)])
            # Also update the model's environment
            self.model.set_env(self.env)

    return ImprovedMutationAgent(model, index_to_action, data["wt_seq"], simulation_result_json_path)
