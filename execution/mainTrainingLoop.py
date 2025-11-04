#!/usr/bin/env python3
"""
Corrected main training loop that properly uses the refactored agent RL policy.
This version handles the initial agent creation without simulation results and updates
the agent after the first episode.
"""

import sys
import shutil
import json
import csv
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Optional
import paramiko
import time
import datetime

# Fix import paths to work from Execution directory
sys.path.append(str(Path(__file__).parent.parent))

from rewardCalculation import reward_calculator
from agentRLPolicyMutation import create_mutation_agent

# Paths - fix to work from Execution directory context
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "Execution" / "results"

REMOTE_USER = "mtelikep"
REMOTE_HOST = "carya.rcdc.uh.edu"
REMOTE_DIR = "project/zerze/mtelikep/SFProject"

ALLOWED_REMOTE_FILES = {
    "Dependencies_and_scripts",
    "seqToPDB.py", "seq2pdb.batch",
}

def upload_files_to_carya(
    file_paths: List[str],
    remote_user: str = "mtelikep",
    remote_host: str = "carya.rcdc.uh.edu",
    remote_dir: str = "/project/zerze/mtelikep/SFProject/",
    ssh_key_path: Optional[str] = None,
    delete_after_upload: bool = False,
):
    """
    Uploads a list of local files to the specified directory on Carya HPC using scp.
    Optionally deletes each file after successful upload.
    """
    for path in file_paths:
        local_file = Path(path).expanduser().resolve()

        if not local_file.exists():
            print(f"[main error] File not found: {local_file}")
            continue

        scp_cmd = ["scp"]
        if ssh_key_path:
            scp_cmd += ["-i", str(Path(ssh_key_path).expanduser())]

        scp_cmd += [str(local_file), f"{remote_user}@{remote_host}:{remote_dir}"]

        print(f"[main] Uploading {local_file.name} → {remote_host}:{remote_dir}")

        try:
            subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
            print(f"[main] Uploaded: {local_file.name}")

            if delete_after_upload:
                try:
                    local_file.unlink()
                    print(f"[main]  Deleted local file: {local_file}")
                except Exception as del_err:
                    print(f"[main warning]  Upload succeeded but failed to delete {local_file}: {del_err}")
        except subprocess.CalledProcessError as e:
            print(f"[main error] Failed to upload {local_file.name}:\n{e.stderr}")

def download_from_carya(
        remote_files: List[str],
        remote_user: str = "mtelikep",
        remote_host: str = "carya.rcdc.uh.edu",
        remote_dir: str = "/project/zerze/mtelikep/SFProject/",
        ssh_key_path: Optional[str] = None,
    ):
    """
    Downloads a list of files from the specified Carya HPC directory to the local current directory.
    """
    for filename in remote_files:
        remote_path = f"{remote_user}@{remote_host}:{remote_dir.rstrip('/')}/{filename}"
        local_path = Path.cwd() / filename

        scp_cmd = ["scp"]
        if ssh_key_path:
            scp_cmd += ["-i", str(Path(ssh_key_path).expanduser())]

        scp_cmd += [remote_path, str(local_path)]

        print(f"⬇️  Downloading {filename} from {remote_host}:{remote_dir}")

        try:
            subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
            print(f"[mdynamics success] Downloaded: {filename} → {local_path}")
        except subprocess.CalledProcessError as e:
            print(f"[mdynamics error] Failed to download {filename}:\n{e.stderr}")

REMOTE_USER = "mtelikep"
REMOTE_HOST = "carya.rcdc.uh.edu"
BASE_PATH = "/project/zerze/mtelikep/SFProject/"

def log(msg: str):
    """Print message with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def open_ssh_session():
    """Open persistent SSH session using paramiko."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, username=REMOTE_USER)  # add key_filename if needed
    log("[SSH INFO] SSH session opened.")
    return client

def close_ssh_session(client):
    """Close persistent SSH session."""
    client.close()
    log("[SSH INFO] SSH session closed.")

def ssh_exec(client, command: str) -> str:
    """Run a command over an open paramiko session."""
    log(f"[SSH DEBUG] Executing remote command: {command}")
    stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        log(f"[SSH WARN] stderr: {err}")
        log(f"[SSH DEBUG] stdout: {out}")
        return out

def wait_for_remote_flag(client, flag_path: str, poll_interval: int = 600):
    """
    Poll the remote HPC until the flag file exists.
    Timestamped logging for every poll attempt (every 10 minutes).
    """
    log(f"[WAIT DEBUG] Start polling for flag: {flag_path}")
    attempt = 0
    while True:
        attempt += 1
        output = ssh_exec(client, f'test -f "{flag_path}" && echo FOUND || echo MISSING').strip()
        log(f"[WAIT DEBUG] Attempt {attempt}: Remote flag status: {output}")
        if output == "FOUND":
            log(f"[WAIT SUCCESS] Flag {flag_path} found on attempt {attempt}!")
            return
        log(f"[WAIT INFO] Flag {flag_path} not yet found, sleeping {poll_interval}s...")
        time.sleep(poll_interval)

def run_batch_and_wait(client, batch_name: str, timeout_hours: int = 10, poll_interval: int = 600):
    """
    Submit a batch job and wait for completion using remote flag polling.
    Aborts if timeout is reached. Polling is every 10 minutes by default.
    """
    def submit_job(batch, extra_export=None):
        export_arg = f"--export={extra_export}" if extra_export else ""
        ssh_cmd = f'cd {BASE_PATH} && sbatch {export_arg} {batch}'
        log(f"[SUBMIT DEBUG] Preparing to submit: {batch} {('(with ' + extra_export + ')' if extra_export else '')}")
        submit_output = ssh_exec(client, ssh_cmd)
        log(f"[SUBMIT DEBUG] sbatch output: {submit_output.strip()}")
        for line in submit_output.splitlines():
            log(f"[SUBMIT DEBUG] Parsing line: {line}")
            if "Submitted batch job" in line:
                job_id = line.strip().split()[-1]
                log(f"[SUBMIT SUCCESS] Job submitted: {job_id}")
                return job_id
        raise RuntimeError(f"[SUBMIT ERROR] Could not parse job ID from sbatch output: {submit_output}")

    log(f"[RUN DEBUG] Starting batch submission sequence for: {batch_name}")

    try:
        job_id = submit_job(batch_name)
        log(f"[RUN INFO] Job ID: {job_id}")
    except Exception as e:
        log(f"[RUN ERROR] Failed to submit job: {e}")
        return

    # Watch flag
    flag_file = f"{BASE_PATH}/output_done_{job_id}.flag"
    log(f"[RUN DEBUG] Waiting for flag: {flag_file}")

    start_time = time.time()
    attempt = 0
    while True:
        attempt += 1
        elapsed = time.time() - start_time
        if elapsed > timeout_hours * 3600:
            log(f"[RUN ERROR] Timeout reached ({timeout_hours}h). Aborting job.")
            ssh_exec(client, f"scancel {job_id}")
            raise TimeoutError(f"Job {job_id} did not finish within {timeout_hours} hours")

        out_flag = ssh_exec(client, f'test -f "{flag_file}" && echo FOUND || echo MISSING').strip()
        log(f"[RUN DEBUG] Attempt {attempt}: flag={out_flag}")
        if out_flag == "FOUND":
            log(f"[RUN SUCCESS] Job {job_id} finished (flag found).")
            return

        log(f"[RUN INFO] Flag not found, sleeping {poll_interval}s...")
        time.sleep(poll_interval)

def plot_scores(history, save_path=RESULTS_DIR / "score_plot.png"):
    """
    Plot the functionality and aggregation scores over episodes.
    """
    episodes = list(range(1, len(history) + 1))
    func_scores = [h["reward"]["mutant_scores"]["functionality_score"] for h in history]
    agg_scores = [h["reward"]["mutant_scores"]["aggregation_score"] for h in history]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, func_scores, 'b-o', label='Functionality Score')
    plt.xlabel('Episode')
    plt.ylabel('Functionality Score')
    plt.title('Functionality Score Over Episodes')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, agg_scores, 'r-o', label='Aggregation Score')
    plt.xlabel('Episode')
    plt.ylabel('Aggregation Score')
    plt.title('Aggregation Score Over Episodes')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Score plot saved to: {save_path}")

def main():
    """
    Main training loop that creates an agent and runs multiple episodes.
    """
    # Ensure directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    MAX_EPISODES = 20  # Adjust as needed
    mutable_positions = str(DATA_DIR / "mutable_positions.json")
    
    # Initialize history
    history = []
    
    print("🚀 Starting mutation training loop...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Results directory: {RESULTS_DIR}")
    print(f"🎯 Mutable positions: {mutable_positions}")
    
    # Create agent without simulation results initially
    # The agent will use fallback scores for the first episode
    print("🚀 Creating mutation agent without simulation results...")
    agent = create_mutation_agent(
        mutable_positions_path=mutable_positions,
        reference_sequence_fasta=str(DATA_DIR / "wt_a_syn.fasta")
        # Note: simulation_result_json_path is not provided initially
    )
    print("✅ Agent created successfully")

    # Main training loop
    for episode in range(MAX_EPISODES):
        print(f"\n=== EPISODE {episode + 1} ===")
        num_loop = f"loop_{episode+1:02d}"
        
        # Select and apply mutation
        print("🎯 Generating mutation...")
        mutation = agent.select_mutation(history)
        mut_id = "_".join(f"{pos}{aa}" for pos, aa in mutation)
        mutated_sequence = agent.apply_mutation(mutation)
        print(f"✅ Generated mutation: {mutation}")
        print(f"🧬 Mutated sequence: {mutated_sequence}")
        
        # Create FASTA file
        fasta_path = Path(f"protein.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">mut_{num_loop}\n{mutated_sequence}\n")

        # Send to Carya & run AlphaFold
        print("📤 Uploading to Carya...")
        upload_files_to_carya([str(fasta_path)])
        client = open_ssh_session()
        run_batch_and_wait(client, "seq2pdb.batch")

        # Retrieve predicted structure
        print("📥 Downloading alphafold results...")
        download_from_carya(["protein/ranked0.pdb"])
        pdb_file = BASE_DIR / "ranked0.pdb"

        # Run simulations (remote process assumed to handle execution)
        print("🔄 Running simulations...")
        def run_sim_pipeline(num_loop = num_loop, pdb_file = pdb_file, fasta_path = fasta_path):
            result = subprocess.run(
                [sys.executable, BASE_DIR / "allSimulationWrapper.py", num_loop, pdb_file, fasta_path],
                capture_output=True,
                text=True
            )
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        run_sim_pipeline()

        # Retrieve all results except mainTrainingLoop.ipynb
        local_result_dir = RESULTS_DIR / num_loop

        def local_cleanup():
            pass
        # Evaluate reward
        sim_result_path = local_result_dir / "simulation_results.json"
        try:
            print("🧮 Calculating reward...")
            reward_result = reward_calculator(
                mutant_json_path=str(sim_result_path),
                wt_json_path=str(DATA_DIR / "wt_scores.json")
            )
            # Extract the scores from the nested structure returned by reward_calculator
            functionality_score = reward_result['mutant_scores']['functionality_score']
            aggregation_score = reward_result['mutant_scores']['aggregation_score']
            print(f"✅ Reward calculated: Func={functionality_score:.3f}, Agg={aggregation_score:.3f}")
        except Exception as e:
            print(f"❌ Reward calculation failed for {num_loop}: {e}")
            continue

        # Update agent
        history.append({"mutation": mutation, "reward": reward_result})
        
        # After first episode, update the agent's simulation path for future episodes
        if episode == 0:
            print(f"🔄 Updating agent with simulation path: {sim_result_path}")
            agent.update_simulation_path(str(sim_result_path))
        
        # Learn from history
        print("🧠 Learning from history...")
        agent.learn_from_history(history)

    # After loop: analyze overall performance
    print("\n=== Training complete ===")
    
    # Save outputs - fix the score access to match the actual structure
    sorted_history = sorted(history, key=lambda x: (x["reward"]["mutant_scores"]["functionality_score"], -x["reward"]["mutant_scores"]["aggregation_score"]), reverse=True)
    best_mutations = [entry for entry in sorted_history if entry["reward"]["mutant_scores"]["functionality_score"] >= 0.90 and entry["reward"]["mutant_scores"]["aggregation_score"] <= 0.20]

    with open(RESULTS_DIR / "best_mutations.json", "w") as f:
        json.dump(best_mutations, f, indent=2)

    with open(RESULTS_DIR / "full_mutation_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(RESULTS_DIR / "mutation_summary.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "mutation", "functionality_score", "aggregation_score"])
        writer.writeheader()
        for i, entry in enumerate(history):
            writer.writerow({
                "episode": i+1,
                "mutation": str(entry["mutation"]),
                "functionality_score": entry["reward"]["mutant_scores"]["functionality_score"],
                "aggregation_score": entry["reward"]["mutant_scores"]["aggregation_score"]
            })

    plot_scores(history)
    print("✅ Results exported.")
    
    # Print summary - fix the score access
    print(f"\n📊 Training Summary:")
    print(f"   Total episodes: {len(history)}")
    print(f"   Best functionality score: {max(h['reward']['mutant_scores']['functionality_score'] for h in history):.3f}")
    print(f"   Best aggregation score: {min(h['reward']['mutant_scores']['aggregation_score'] for h in history):.3f}")
    print(f"   Successful mutations (func≥0.9, agg≤0.2): {len(best_mutations)}")

if __name__ == "__main__":
    main()
