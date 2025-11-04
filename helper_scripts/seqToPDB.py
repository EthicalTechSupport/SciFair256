# CREATED BY MANAS TELIKEPALLI

import subprocess
import time
from pathlib import Path
import argparse
import paramiko
from datetime import datetime

REMOTE_USER = "mtelikep"
REMOTE_HOST = "carya.rcdc.uh.edu"
BASE_PATH = "/project/zerze/mtelikep/SFProject/"
LOCAL_PATH = "/mnt/c/Users/mteli/SFproject256/SFProject256/"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def open_ssh_session():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, username=REMOTE_USER)
    log("[SSH INFO] SSH session opened.")
    return client

def close_ssh_session(client):
    client.close()
    log("[SSH INFO] SSH session closed.")

def ssh_exec(client, command: str) -> str:
    log(f"[SSH DEBUG] Executing remote command: {command}")
    stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        log(f"[SSH WARN] stderr: {err}")
    return out

def run_seq2pdb_pipeline(local_fasta: str):
    client = open_ssh_session()
    try:
        fasta_path = Path(local_fasta).resolve()
        if not fasta_path.exists():
            raise FileNotFoundError(f"[❌] FASTA file not found: {fasta_path}")

        # Step 1: Upload fasta to Carya
        rsync_upload = [
            "rsync", "-avz",
            str(fasta_path),
            f"{REMOTE_USER}@{REMOTE_HOST}:{BASE_PATH}/seq2pdb.fasta"
        ]
        log(f"[UPLOAD INFO] Uploading FASTA: {' '.join(rsync_upload)}")
        subprocess.run(rsync_upload, check=True)
        log("[UPLOAD SUCCESS] FASTA uploaded to Carya.")

        # Step 2: Submit batch
        ssh_cmd = f"cd {BASE_PATH} && sbatch seq2pdb.batch"
        out = ssh_exec(client, ssh_cmd)
        log(f"[SUBMIT INFO] sbatch output: {out}")

        # Step 3: Wait for output_done.flag
        flag_path = f"{BASE_PATH}/seq2pdb/output_done.flag"
        log(f"[WAIT INFO] Polling for flag: {flag_path}")
        while True:
            result = ssh_exec(client, f'test -f "{flag_path}" && echo FOUND || echo MISSING')
            if result.strip() == "FOUND":
                log("[WAIT SUCCESS] output_done.flag found.")
                break
            log("[WAIT INFO] Flag not found. Sleeping 600s...")
            time.sleep(600)

        # Step 4: Rsync ranked_0.pdb back to local
        rsync_download = [
            "rsync", "-avz",
            "--include", "ranked_0.pdb",
            "--exclude", "*",
            f"{REMOTE_USER}@{REMOTE_HOST}:{BASE_PATH}/seq2pdb/",
            LOCAL_PATH
        ]
        log(f"[RSYNC INFO] Downloading ranked_0.pdb: {' '.join(rsync_download)}")
        subprocess.run(rsync_download, check=True)
        log("[RSYNC SUCCESS] ranked_0.pdb downloaded.")

        # Step 5: Cleanup seq2pdb directory on Carya
        cleanup_cmd = f"rm -rf {BASE_PATH}/seq2pdb"
        ssh_exec(client, cleanup_cmd)
        log("[CLEANUP SUCCESS] Deleted remote /seq2pdb/ directory.")

    finally:
        close_ssh_session(client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run seq2pdb pipeline on Carya and retrieve ranked_0.pdb.")
    parser.add_argument("fasta_path", help="Path to input FASTA file (seq2pdb.fasta)")
    args = parser.parse_args()

    run_seq2pdb_pipeline(args.fasta_path)
