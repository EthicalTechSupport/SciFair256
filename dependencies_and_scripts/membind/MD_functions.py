# CREATED BY MANAS TELIKEPALLI

from pathlib import Path
import subprocess
import numpy as np
import MDAnalysis as mda
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array
import re
import subprocess
from typing import List, Literal, Optional, Tuple
import time
import os
import argparse
import shutil
import time
from datetime import datetime

def run_mdynamics_pipeline(pdb_file, output_prefix):
    def run_batch_and_wait(batch_name):
        remote = "mtelikep@carya.rcdc.uh.edu"
        base_path = "/project/zerze/mtelikep/SFProject/"

        def submit_job(batch):
            ssh_cmd = f"ssh {remote} 'cd {base_path} && sbatch {batch}'"
            print(f"[[mdynamics] Submitting] {batch}")
            submit_output = subprocess.check_output(ssh_cmd, shell=True, text=True)
            print(f"[[mdynamics] Submit Output] {submit_output.strip()}")
            for line in submit_output.splitlines():
                if "Submitted batch job" in line:
                    return line.strip().split()[-1]
            raise RuntimeError(f"[mdynamics error] Could not parse job ID from sbatch output: {submit_output}")

        # Submit nogpu first, then normal
        nogpu_batch = batch_name.replace(".batch", "_nogpu.batch")
        nogpu_job_id = submit_job(nogpu_batch)
        time.sleep(2)
        normal_job_id = submit_job(batch_name)

        submit_time = datetime.now()
        normal_started = False

        while True:
            # Get short-form status: R = running, PD = pending
            ssh_cmd = f"ssh {remote} 'squeue -h -o \"%A %t\" -j {nogpu_job_id},{normal_job_id}'"
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
            jobs_status = {line.split()[0]: line.split()[1] for line in result.stdout.strip().splitlines()}

            # If normal starts <2h after submission → cancel nogpu
            if jobs_status.get(normal_job_id) == "R":
                if not normal_started:
                    elapsed = (datetime.now() - submit_time).total_seconds()
                    if elapsed < 7200:
                        print(f"[[mdynamics] Normal {normal_job_id} started in {elapsed/60:.1f} min, cancelling nogpu {nogpu_job_id}]")
                        subprocess.run(f"ssh {remote} 'scancel {nogpu_job_id}'", shell=True)
                    normal_started = True

            # If 2h pass and normal still pending → cancel normal, keep nogpu
            elif jobs_status.get(normal_job_id) == "PD":
                if not normal_started and (datetime.now() - submit_time).total_seconds() >= 7200:
                    print(f"[[mdynamics] Normal {normal_job_id} not started after 2h, cancelling normal and keeping nogpu]]")
                    subprocess.run(f"ssh {remote} 'scancel {normal_job_id}'", shell=True)
                    normal_started = True

            # Exit when both jobs are gone
            ssh_cmd = f"ssh {remote} 'squeue -h -o \"%A\" -j {nogpu_job_id},{normal_job_id}'"
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
            active_jobs = set(result.stdout.strip().splitlines())
            if not (nogpu_job_id in active_jobs or normal_job_id in active_jobs):
                print("[[mdynamics success] Done] Both jobs finished or cancelled.")
                break

            print(f"[[mdynamics] Waiting] Active jobs: {', '.join(active_jobs) if active_jobs else 'None'}")
            time.sleep(30)

    def run_batches_and_wait_parallel(batch_list):
        remote = "mtelikep@carya.rcdc.uh.edu"
        base_path = "/project/zerze/mtelikep/SFProject/"

        def submit_job(batch):
            ssh_cmd = f"ssh {remote} 'cd {base_path} && sbatch {batch}'"
            print(f"[[mdynamics] Submitting] {batch}")
            submit_output = subprocess.check_output(ssh_cmd, shell=True, text=True)
            print(f"[[mdynamics] Submit Output] {submit_output.strip()}")
            for line in submit_output.splitlines():
                if "Submitted batch job" in line:
                    return line.strip().split()[-1]
            raise RuntimeError(f"[mdynamics error] Could not parse job ID from sbatch output: {submit_output}")

        job_pairs = {}  # batch_name -> [nogpu_id, normal_id, submit_time, normal_started]

        for batch_name in batch_list:
            nogpu_batch = batch_name.replace(".batch", "_nogpu.batch")
            nogpu_job_id = submit_job(nogpu_batch)
            time.sleep(2)
            normal_job_id = submit_job(batch_name)
            job_pairs[batch_name] = [nogpu_job_id, normal_job_id, datetime.now(), False]

        while True:
            # Get all job IDs
            all_ids = [jid for pair in job_pairs.values() for jid in pair[:2]]

            # Check status in short form
            ssh_cmd = f"ssh {remote} 'squeue -h -o \"%A %t\" -j {','.join(all_ids)}'"
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
            jobs_status = {line.split()[0]: line.split()[1] for line in result.stdout.strip().splitlines()}

            for batch_name, (nogpu_id, normal_id, submit_time, normal_started) in job_pairs.items():
                # If normal starts <2h → cancel nogpu
                if jobs_status.get(normal_id) == "R":
                    if not normal_started:
                        elapsed = (datetime.now() - submit_time).total_seconds()
                        if elapsed < 7200:
                            print(f"[[mdynamics] Normal {batch_name} job {normal_id} started in {elapsed/60:.1f} min, cancelling nogpu {nogpu_id}]")
                            subprocess.run(f"ssh {remote} 'scancel {nogpu_id}'", shell=True)
                        job_pairs[batch_name][3] = True

                # If 2h pass and still pending → cancel normal
                elif jobs_status.get(normal_id) == "PD":
                    if not normal_started and (datetime.now() - submit_time).total_seconds() >= 7200:
                        print(f"[[mdynamics] Normal {batch_name} job {normal_id} not started after 2h, cancelling normal and keeping nogpu]]")
                        subprocess.run(f"ssh {remote} 'scancel {normal_id}'", shell=True)
                        job_pairs[batch_name][3] = True

            # Check if all jobs are gone
            ssh_cmd = f"ssh {remote} 'squeue -h -o \"%A\" -j {','.join(all_ids)}'"
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
            active_jobs = set(result.stdout.strip().splitlines())
            if not active_jobs:
                print("[[mdynamics success] Done] All jobs finished or cancelled.")
                break

            print(f"[[mdynamics] Waiting] Active jobs: {', '.join(active_jobs) if active_jobs else 'None'}")
            time.sleep(30)

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

        Args:
            file_paths: List of file paths to upload (absolute or relative).
            remote_user: Your Carya username.
            remote_host: HPC hostname (default is Carya).
            remote_dir: Target directory on the HPC.
            ssh_key_path: Optional path to your private SSH key (e.g., ~/.ssh/id_ed25519).
            delete_after_upload: If True, delete each local file after successful upload.
        """
        for path in file_paths:
            local_file = Path(path).expanduser().resolve()

            if not local_file.exists():
                print(f"[mdynamics error] File not found: {local_file}")
                continue

            scp_cmd = ["scp"]
            if ssh_key_path:
                scp_cmd += ["-i", str(Path(ssh_key_path).expanduser())]

            scp_cmd += [str(local_file), f"{remote_user}@{remote_host}:{remote_dir}"]

            print(f"[mdynamics] Uploading {local_file.name} → {remote_host}:{remote_dir}")

            try:
                subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
                print(f"[mdynamics] Uploaded: {local_file.name}")

                if delete_after_upload:
                    try:
                        local_file.unlink()
                        print(f"[mdynamics]  Deleted local file: {local_file}")
                    except Exception as del_err:
                        print(f"[mdynamics warning]  Upload succeeded but failed to delete {local_file}: {del_err}")
            except subprocess.CalledProcessError as e:
                print(f"[mdynamics error] Failed to upload {local_file.name}:\n{e.stderr}")

    def download_from_carya(
            remote_files: List[str],
            remote_user: str = "mtelikep",
            remote_host: str = "carya.rcdc.uh.edu",
            remote_dir: str = "/project/zerze/mtelikep/SFProject/",
            ssh_key_path: Optional[str] = None,
        ):
        """
        Downloads a list of files from the specified Carya HPC directory to the local current directory.
        
        Args:
            remote_files: List of filenames (not paths) to download from the remote_dir.
            remote_user: Your Carya username.
            remote_host: HPC hostname (default is Carya).
            remote_dir: Directory on the HPC where files are located.
            ssh_key_path: Optional path to your private SSH key (e.g., ~/.ssh/id_ed25519).
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

    def run_rsync(args: List[str]):
        """
        Runs an rsync command with arbitrary arguments.

        Example:
            run_rsync([
                "-avz",
                "--include", "umbrella_win_*.batch",
                "--exclude", "*",
                str(Path.cwd()),
                "mtelikep@carya.rcdc.uh.edu:/project/zerze/mtelikep/SFProject/"
            ])

        Args:
            args: List of arguments to pass directly to rsync.
        """
        cmd = ["rsync"] + args
        print(f"🚀 Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print("✅ rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ rsync failed with return code {e.returncode}")

    def run_martinize_and_convert_gro_wsl(
        pdb_file,
        output_prefix="aSyn",
        martini_ff="martini3001",
        ef=500,
        el=0.5,
        eu=1.0,
        use_elastic=True,
        ignore_hydrogens=True
    ):
        """
        Runs martinize2 and converts the resulting CG structure to GRO format using GROMACS.
        All commands are executed via WSL. Returns WSL paths to .gro and .top files.
        """

        def run_wsl_command(cmd):
            print(f"🔧 Running WSL Command:\n{cmd}")
            result = subprocess.run(f"wsl bash -c \"{cmd}\"", shell=True, capture_output=True, text=True)
            print("📤 STDOUT:\n", result.stdout)
            print("📥 STDERR:\n", result.stderr)
            if result.returncode != 0:
                raise RuntimeError("❌ Command failed")
            return result

        # Convert to WSL path
        cwd_wsl = Path.cwd().as_posix().replace("C:", "/mnt/c")
        pdb_file_wsl = str(Path(pdb_file).resolve()).replace("C:", "/mnt/c").replace("\\", "/")
        
        # Output file paths (WSL-accessible)
        wsl_output_top = f"{cwd_wsl}/{output_prefix}.top"
        wsl_output_pdb = f"{cwd_wsl}/{output_prefix}_cg.pdb"
        wsl_output_gro = f"{cwd_wsl}/{output_prefix}_cg.gro"

        # Step 1: Run martinize2
        cmd_martinize = (
            f"source ~/martini-env/bin/activate && "
            f"martinize2 -f {pdb_file_wsl} -o {wsl_output_top} -x {wsl_output_pdb} "
            f"-ff {martini_ff} -name Protein -p backbone"
        )
        if use_elastic:
            cmd_martinize += f" -elastic -ef {ef} -el {el} -eu {eu}"
        if ignore_hydrogens:
            cmd_martinize += " -ignh"

        run_wsl_command(cmd_martinize)
        print(f"✅ CG PDB and TOP generated: {output_prefix}_cg.pdb, {output_prefix}.top")

        # Step 2: Convert to GRO
        cmd_pdb2gro = (
            f"source ~/martini-env/bin/activate && "
            f"gmx editconf -f {wsl_output_pdb} -o {wsl_output_gro}"
        )
        run_wsl_command(cmd_pdb2gro)
        print(f"✅ Converted to GRO: {output_prefix}_cg.gro")

        return wsl_output_gro, wsl_output_top

    def reposition_and_rotate_protein(
        protein_gro_in,
        protein_gro_out,
        bilayer_box,
        upper_leaflet_z,
        translate_z=4.0
    ):
        """
        Reads a .gro file of a CG protein, aligns its membrane-binding face to the Z axis,
        centers it in XY above a bilayer, and translates it above the upper leaflet.
        """

        def read_gro(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
            title = lines[0].strip()
            natoms = int(lines[1])
            atom_lines = lines[2:2+natoms]
            box_line = list(map(float, lines[2+natoms].split()))
            coords = []
            resids = []
            for line in atom_lines:
                resnum_str = line[0:5].strip()
                if not resnum_str.isdigit():
                    raise ValueError(f"Residue number parse error: '{resnum_str}' in line: {line}")
                resids.append(int(resnum_str))
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                coords.append([x, y, z])
            coords = np.array(coords)
            resids = np.array(resids)
            return title, natoms, atom_lines, coords, resids, box_line

        def write_gro(filename, title, natoms, atom_lines, coords, box):
            with open(filename, 'w') as f:
                f.write(f"{title}\n")
                f.write(f"{natoms}\n")
                for i, line in enumerate(atom_lines):
                    new_line = line[:20] + f"{coords[i,0]:8.3f}{coords[i,1]:8.3f}{coords[i,2]:8.3f}" + line[44:]
                    f.write(new_line)
                f.write(f"{box[0]:9.5f} {box[1]:9.5f} {box[2]:9.5f}\n")

        def compute_rotation_matrix(normal, target=[0, 0, 1]):
            normal = normal / np.linalg.norm(normal)
            target = np.array(target) / np.linalg.norm(target)
            v = np.cross(normal, target)
            c = np.dot(normal, target)
            if np.allclose(c, 1):
                return np.identity(3)
            if np.allclose(c, -1):
                return -np.identity(3)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.identity(3) + vx + (vx @ vx) * ((1 - c) / (np.linalg.norm(v)**2))
            return R

        # Step 1: Read GRO
        title, natoms, atom_lines, coords, resids, _ = read_gro(protein_gro_in)

        # Step 2: Identify membrane-binding face (residues 1–60)
        binding_mask = (resids >= 1) & (resids <= 60)
        binding_coords = coords[binding_mask]

        # Step 3: PCA normal vector from binding face
        binding_centered = binding_coords - np.mean(binding_coords, axis=0)
        _, _, vh = np.linalg.svd(binding_centered)
        normal = vh[2]

        # Step 4: Align normal to Z axis
        R = compute_rotation_matrix(normal, target=[0, 0, 1])
        coords = coords @ R.T

        # Step 5: Center in XY
        protein_center_xy = np.mean(coords[:, :2], axis=0)
        bilayer_center_xy = np.array([bilayer_box[0]/2, bilayer_box[1]/2])
        translation_xy = bilayer_center_xy - protein_center_xy
        coords[:, 0] += translation_xy[0]
        coords[:, 1] += translation_xy[1]

        # Step 6: Lift above bilayer surface
        min_z = np.min(coords[:, 2])
        coords[:, 2] += (upper_leaflet_z + translate_z) - min_z

        # Step 7: Write new GRO file
        write_gro(protein_gro_out, title, natoms, atom_lines, coords, bilayer_box)
        print(f"✅ Protein rotated, positioned, and saved to {protein_gro_out}")

    def setup_membrane_system_wsl(
        protein_gro_in: str,
        output_gro: str,
        output_top: str,
        target_protein_center_z: float = 5.0,
        box_x: float = 25.0,
        box_y: float = 25.0,
        box_z: float = 50.0,
        insane_path: str = "~/martini-env/bin/insane",
        martini_itp_path: str = "./Dependencies_and_scripts/mdynamics/martini_v300/martini_v300/martini_v3.0.0.itp"
    ):
        """
        Shifts protein to specified Z position and generates a solvated membrane system with INSANE.
        """
        
        def read_gro(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
            title = lines[0].rstrip()
            natoms = int(lines[1])
            atom_lines = lines[2:2+natoms]
            box_line = lines[2+natoms].split()
            coords = []
            for line in atom_lines:
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                coords.append([x, y, z])
            coords = np.array(coords)
            return title, natoms, atom_lines, coords, box_line

        def write_gro(filename, title, natoms, atom_lines, coords, box_line):
            with open(filename, 'w') as f:
                f.write(f"{title}\n")
                f.write(f"{natoms}\n")
                for i, line in enumerate(atom_lines):
                    new_line = line[:20] + f"{coords[i,0]:8.3f}{coords[i,1]:8.3f}{coords[i,2]:8.3f}" + line[44:]
                    f.write(new_line)
                f.write(" ".join(box_line) + "\n")

        # Convert paths to WSL
        cwd_wsl = Path.cwd().as_posix().replace("C:", "/mnt/c")
        protein_gro_wsl = f"{cwd_wsl}/{protein_gro_in}"
        output_gro_wsl = f"{cwd_wsl}/{output_gro}"
        output_top_wsl = f"{cwd_wsl}/{output_top}"
        
        # Step 1: Shift protein to desired Z center
        shifted_protein_gro = "temp_shifted.gro"
        shifted_protein_gro_wsl = f"{cwd_wsl}/{shifted_protein_gro}"
        
        title, natoms, atom_lines, coords, box_line = read_gro(protein_gro_in)
        current_center_z = coords[:, 2].mean()
        shift = target_protein_center_z - current_center_z
        coords[:, 2] += shift
        write_gro(shifted_protein_gro, title, natoms, atom_lines, coords, box_line)
        print(f"✅ Protein shifted to Z = {target_protein_center_z:.2f} nm (center)")

        # Step 2: Run INSANE with shifted protein
        lipid_flags = "-l DOPC:20 -l DOPE:50 -l DOPS:30"
        command = (
            f"source ~/martini-env/bin/activate && "
            f"{insane_path} -f {shifted_protein_gro_wsl} -o {output_gro_wsl} -p {output_top_wsl} "
            f"-x {box_x} -y {box_y} -z {box_z} "
            f"{lipid_flags} -sol W -salt 0.15 -m {martini_itp_path}"
        )
        wsl_command = f"wsl bash -c \"{command}\""
        print("🚀 Running INSANE in WSL...")
        result = subprocess.run(wsl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print("✅ INSANE membrane+protein system generated!")
            print(result.stdout)
        else:
            print("❌ INSANE error:")
            print(result.stderr)
            
        # Clean up temporary file
        if os.path.exists(shifted_protein_gro):
            os.remove(shifted_protein_gro)

    def convert_top_to_martini3(top_file: str | Path):
        """
        Converts a GROMACS topology file to Martini 3 by:
        - Removing Martini 2 includes
        - Replacing all instances of 'Protein' with 'Protein_0'
        - Inserting the correct Martini 3 include statements at the top
        - Backing up the original file as .top.bak

        Parameters
        ----------
        top_file : str or Path
            Path to the original .top file to convert
        """
        top_path = Path(top_file).resolve()

        if not top_path.exists():
            raise FileNotFoundError(f"❌ Topology file not found: {top_path}")

        # Martini 3 includes
        martini3_includes = [
            '#include "./Dependencies_and_scripts/mdynamics/martini_v300/martini_v300/martini_v3.0.0.itp"',
            '#include "./Dependencies_and_scripts/mdynamics/martini_v300/martini_v300/martini_v3.0.0_phospholipids_v1.itp"',
            '#include "./Dependencies_and_scripts/mdynamics/martini_v300/martini_v300/martini_v3.0.0_solvents_v1.itp"',
            '#include "./Dependencies_and_scripts/mdynamics/martini_v300/martini_v300/martini_v3.0.0_ions_v1.itp"',
            '#include "Protein_0.itp"'
        ]

        # Read original file
        with open(top_path, "r") as f:
            lines = f.readlines()

        # Remove Martini 2 includes and replace 'Protein' with 'Protein_0'
        filtered = []
        for line in lines:
            if "martini.itp" in line or "martini_v2" in line:
                continue
            filtered.append(line.replace("Protein", "Protein_0"))

        # Insert Martini 3 includes
        final_lines = martini3_includes + ["\n"] + filtered

        # Backup original
        backup_path = top_path.with_suffix(".top.bak")
        top_path.rename(backup_path)

        # Write updated file
        with open(top_path, "w") as f:
            for line in final_lines:
                f.write(line if line.endswith("\n") else line + "\n")

        print(f"✅ Updated {top_path.name} for Martini 3.")
        print(f"📦 Backup saved as {backup_path.name}")

    def make_index_file(gro_file, ndx_file="index.ndx"):
        # Convert paths to WSL
        cwd_wsl = Path.cwd().as_posix().replace("C:", "/mnt/c")
        gro_file_wsl = f"{cwd_wsl}/{gro_file}"
        ndx_file_wsl = f"{cwd_wsl}/{ndx_file}"
        
        cmd = (
            f"source ~/martini-env/bin/activate && "
            f"echo -e 'r W | r NA+ | r CL-\\n"
            f"name 22 Solvent\\n"
            f"r DOPC | r DOPE | r DOPS\\n"
            f"name 23 Membrane\\n"
            f"!r W & !r NA+ & !r CL- & !r DOPC & !r DOPE & !r DOPS\\n"
            f"name 24 Protein\\n"
            f"q' | gmx make_ndx -f {gro_file_wsl} -o {ndx_file_wsl}"
        )
        print("🧬 Creating index.ndx with manual selection...")
        subprocess.run(["wsl", "-e", "bash", "-c", cmd], check=True)

    def rename_ndx_groups(input_ndx, gro_file, output_ndx):
        # Convert paths to WSL
        cwd_wsl = Path.cwd().as_posix().replace("C:", "/mnt/c")
        input_ndx_wsl = f"{cwd_wsl}/{input_ndx}"
        gro_file_wsl = f"{cwd_wsl}/{gro_file}"
        output_ndx_wsl = f"{cwd_wsl}/{output_ndx}"
        
        bash_cmd = (
            f"source ~/martini-env/bin/activate && "
            f"echo 'name 20 Solvent\nname 21 Membrane\nname 22 Protein\nq' | "
            f"gmx make_ndx -f {gro_file_wsl} -n {input_ndx_wsl} -o {output_ndx_wsl}"
        )
        print("🧬 Renaming index.ndx ...")
        subprocess.run(["wsl", "-e", "bash", "-c", bash_cmd], check=True)

    def add_upper_leaflet_group_clustered(
        gro_file="system_solvated.gro",
        ndx_file="index_named.ndx",
        group_name="Upper_PO4"
    ):
        u = mda.Universe(gro_file)
        PO4 = u.select_atoms("name PO4")
        z_coords = PO4.positions[:, 2].reshape(-1, 1)

        # Cluster PO4 atoms into 2 groups (leaflets)
        kmeans = KMeans(n_clusters=2, n_init='auto').fit(z_coords)
        labels = kmeans.labels_

        # Determine which cluster is upper
        mean_z = [np.mean(z_coords[labels == i]) for i in range(2)]
        upper_label = np.argmax(mean_z)

        upper_leaflet = PO4[labels == upper_label]
        atom_ids = upper_leaflet.ids

        print(f"✅ Selected {len(atom_ids)} PO4 atoms in upper leaflet (clustered method)")

        # Append to index file
        with open(ndx_file, "a") as f:
            f.write(f"\n[ {group_name} ]\n")
            for i, atom_id in enumerate(atom_ids, 1):
                f.write(f"{atom_id} ")
                if i % 15 == 0:
                    f.write("\n")
            f.write("\n")

        print(f"✅ Appended group '{group_name}' to {ndx_file}")

    def run_gromacs_minimization():
        run_batch_and_wait("em.batch")

    def run_nvt_batch():
        run_batch_and_wait("nvt.batch")

    def run_npt_batch():
        run_batch_and_wait("npt.batch")

    def run_pulling_batch():
        run_batch_and_wait("pull.batch")

    def extract_snapshots(
        n_frames=601,
        tpr_file="pull.tpr",
        xtc_file="pull.xtc",
        index_file="index_named.ndx",
        group1="22",
        group2="Upper_PO4",
        env_activate="source ~/martini-env/bin/activate",
        summary_file="summary_distances.dat",
        spacing_threshold=0.2
    ):
        def run_in_wsl(command: str):
            """Run a command in WSL bash shell and stream output."""
            process = subprocess.Popen(
                ["wsl", "-e", "bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in process.stdout:
                print(line, end="")
            process.stdout.close()
            return process.wait()

        # Convert paths to WSL
        cwd_wsl = Path.cwd().as_posix().replace("C:", "/mnt/c")
        tpr_file_wsl = f"{cwd_wsl}/{tpr_file}"
        xtc_file_wsl = f"{cwd_wsl}/{xtc_file}"
        index_file_wsl = f"{cwd_wsl}/{index_file}"

        # Step 1: Generate .gro files
        print("📦 Splitting trajectory into .gro files...")
        cmd_split = f'{env_activate} && cd {cwd_wsl} && echo 0 | gmx trjconv -s {tpr_file_wsl} -f {xtc_file_wsl} -o conf.gro -sep'
        run_in_wsl(cmd_split)

        # Step 2: Compute distances
        for i in range(n_frames):
            gro_file = f"{cwd_wsl}/conf{i}.gro"
            dist_file = f"{cwd_wsl}/dist{i}.xvg"
            print(f"📏 Frame {i}: Computing distance...")
            cmd = (
                f"{env_activate} && cd {cwd_wsl} && "
                f"gmx distance -s {tpr_file_wsl} -f {gro_file} -n {index_file_wsl} "
                f"-select 'com of group {group1} plus com of group \"{group2}\"' -oall {dist_file}"
            )
            run_in_wsl(cmd)

        # Step 3: Compile summary file
        print("📄 Compiling raw summary...")
        with open(summary_file, "w") as summary:
            for i in range(n_frames):
                dist_file = f"dist{i}.xvg"
                try:
                    with open(dist_file, "r") as f:
                        lines = f.readlines()
                    last_line = [l for l in lines if not l.startswith(("#", "@"))][-1]
                    distance = last_line.strip().split()[1]
                    summary.write(f"{i} {distance}\n")
                except Exception as e:
                    print(f"⚠️ Error parsing {dist_file}: {e}")
                finally:
                    if os.path.exists(dist_file):
                        os.remove(dist_file)

        # Step 4: Filter by distance spacing
        print(f"🔍 Filtering by ≥{spacing_threshold} nm spacing...")
        filtered_lines = []
        last_dist = None
        keep_frames = set()

        with open(summary_file, "r") as f:
            for line in f:
                i_str, dist_str = line.strip().split()
                i, dist = int(i_str), float(dist_str)
                if last_dist is None or abs(dist - last_dist) >= spacing_threshold:
                    filtered_lines.append(f"{i} {dist}\n")
                    keep_frames.add(i)
                    last_dist = dist
                else:
                    gro_file = f"conf{i}.gro"
                    if os.path.exists(gro_file):
                        os.remove(gro_file)

        # Step 5: Overwrite summary
        with open(summary_file, "w") as f:
            f.writelines(filtered_lines)

        print(f"✅ Final: {len(filtered_lines)} snapshots saved in {summary_file}")



    def generate_npt_batch_scripts_from_conf_files(
            job_name="NPT",
            account="zerze",
            email="mtelikepalli@gmail.com",
            gro_pattern="conf*.gro"
        ) -> List[str]:
        cwd = Path.cwd()
        gro_files = sorted(cwd.glob(gro_pattern))

        def extract_conf_number(file):
            match = re.match(r"conf(\d+)\.gro", file.name)
            return int(match.group(1)) if match else float("inf")

        gro_files = sorted(gro_files, key=extract_conf_number)

        # Templates
        gpu_template = """#!/bin/bash
#SBATCH -J {job_name}_{index}
#SBATCH -t 9:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -A {account}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN,END,FAIL

module load GROMACS/2023.5-foss-2023a-CUDA-12.1.1-PLUMED-2.9.0

gmx grompp -f ./Dependencies_and_scripts/mdynamics/mdp_files/npt.mdp -c {gro_file} -r {gro_file} -p topol_updated.top -n index_named.ndx -o npt_{index}.tpr -maxwarn 2
gmx mdrun -deffnm npt_{index} -ntomp 8 -v -ntmpi 1
"""

        nogpu_template = """#!/bin/bash
#SBATCH -J {job_name}_{index}_nogpu
#SBATCH -t 9:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -A {account}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN,END,FAIL

module load GROMACS/2023.5-foss-2023a-PLUMED-2.9.0

gmx grompp -f ./Dependencies_and_scripts/mdynamics/mdp_files/npt.mdp -c {gro_file} -r {gro_file} -p topol_updated.top -n index_named.ndx -o npt_{index}.tpr -maxwarn 2
gmx mdrun -deffnm npt_{index} -ntomp 12 -v -ntmpi 1
"""

        batch_paths = []

        for new_index, gro_file in enumerate(gro_files):
            idx_str = f"{new_index:02d}"

            # GPU version
            gpu_batch_contents = gpu_template.format(
                job_name=job_name,
                index=idx_str,
                account=account,
                email=email,
                gro_file=gro_file.name
            )
            gpu_filename = cwd / f"npt_win_{idx_str}.batch"
            with open(gpu_filename, "w", newline="\n") as f:
                f.write(gpu_batch_contents)
            batch_paths.append(gpu_filename.name)
            print(f"📝 Created {gpu_filename.name}")

            # No CPU version
            nogpu_batch_contents = nogpu_template.format(
                job_name=job_name,
                index=idx_str,
                account=account,
                email=email,
                gro_file=gro_file.name
            )
            nogpu_filename = cwd / f"npt_win_{idx_str}_nogpu.batch"
            with open(nogpu_filename, "w", newline="\n") as f:
                f.write(nogpu_batch_contents)
            batch_paths.append(nogpu_filename.name)
            print(f"📝 Created {nogpu_filename.name}")

        try:
            upload_files_to_carya(batch_paths)
        except NameError:
            print("⚠️ upload_files_to_carya is not defined. Skipping upload.")

        return batch_paths


    def generate_umbrella_batch_scripts_from_npt_files(
            job_name="UMBRELLA",
            account="zerze",
            email="mtelikepalli@gmail.com",
            gro_pattern="npt_*.gro"
        ) -> List[str]:
        cwd = Path.cwd()
        gro_files = sorted(cwd.glob(gro_pattern))

        def extract_index(file):
            match = re.match(r"npt_(\d+)\.gro", file.name)
            return int(match.group(1)) if match else float("inf")

        gro_files = sorted(gro_files, key=extract_index)

        gpu_template = """#!/bin/bash
#SBATCH -J {job_name}{index}
#SBATCH -t 9:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -A {account}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN,END,FAIL

module load GROMACS/2023.5-foss-2023a-CUDA-12.1.1-PLUMED-2.9.0

gmx grompp -f ./Dependencies_and_scripts/mdynamics/mdp_files/umbrella.mdp -c {gro_file} -r {gro_file} -p topol_updated.top -n index_named.ndx -o umbrella_{index}.tpr -maxwarn 2
gmx mdrun -deffnm umbrella_{index} -ntomp 8 -v -ntmpi 1
"""

        nogpu_template = """#!/bin/bash
#SBATCH -J {job_name}{index}_nogpu
#SBATCH -t 9:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -A {account}
#SBATCH --mail-user={email}
#SBATCH --mail-type=BEGIN,END,FAIL

module load GROMACS/2023.5-foss-2023a-PLUMED-2.9.0

gmx grompp -f ./Dependencies_and_scripts/mdynamics/mdp_files/umbrella.mdp -c {gro_file} -r {gro_file} -p topol_updated.top -n index_named.ndx -o umbrella_{index}.tpr -maxwarn 2
gmx mdrun -deffnm umbrella_{index} -ntomp 12 -v -ntmpi 1
"""

        batch_paths = []

        for gro_file in gro_files:
            idx_str = f"{extract_index(gro_file):02d}"

            # GPU version
            gpu_batch_contents = gpu_template.format(
                job_name=job_name,
                index=idx_str,
                account=account,
                email=email,
                gro_file=gro_file.name
            )
            gpu_filename = cwd / f"umbrella_win_{idx_str}.batch"
            with open(gpu_filename, "w", newline="\n") as f:
                f.write(gpu_batch_contents)
            batch_paths.append(gpu_filename.name)
            print(f"📝 Created {gpu_filename.name}")

            # No CPU version
            nogpu_batch_contents = nogpu_template.format(
                job_name=job_name,
                index=idx_str,
                account=account,
                email=email,
                gro_file=gro_file.name
            )
            nogpu_filename = cwd / f"umbrella_win_{idx_str}_nogpu.batch"
            with open(nogpu_filename, "w", newline="\n") as f:
                f.write(nogpu_batch_contents)
            batch_paths.append(nogpu_filename.name)
            print(f"📝 Created {nogpu_filename.name}")

        try:
            upload_files_to_carya(batch_paths)
        except NameError:
            print("⚠️ upload_files_to_carya is not defined. Skipping upload.")

        return batch_paths


    def submit_windows_to_npt():
        run_batches_and_wait_parallel(generate_npt_batch_scripts_from_conf_files())

    def submit_windows_to_umbrella():
        run_batches_and_wait_parallel(generate_umbrella_batch_scripts_from_npt_files())

    def create_file_lists(
        tpr_out="tpr-files.dat",
        pullf_out="pullf-files.dat",
        prefix="umbrella",
        directory="."
    ):
        dir_path = Path(directory)

        # Match umbrella_XX.tpr and umbrella_XX_pullf.xvg
        tpr_files = sorted(dir_path.glob(f"{prefix}_*.tpr"))
        pullf_files = sorted(dir_path.glob(f"{prefix}_*_pullf.xvg"))

        # Extract index for sorting and matching
        def extract_index(file, pullf=False):
            pattern = f"{prefix}_(\\d+)_pullf\\.xvg" if pullf else f"{prefix}_(\\d+)\\.tpr"
            match = re.match(pattern, file.name)
            return int(match.group(1)) if match else None

        # Filter and sort by index
        tpr_files = [(extract_index(f), f.name) for f in tpr_files if extract_index(f) is not None]
        pullf_files = [(extract_index(f, pullf=True), f.name) for f in pullf_files if extract_index(f, pullf=True) is not None]

        tpr_files.sort()
        pullf_files.sort()

        # Write to output files
        with open(tpr_out, 'w', newline='\n') as f:
            for _, name in tpr_files:
                f.write(f"{name}\n")
        print(f"📄 Created {tpr_out} with {len(tpr_files)} entries.")

        with open(pullf_out, 'w', newline='\n') as f:
            for _, name in pullf_files:
                f.write(f"{name}\n")
        print(f"📄 Created {pullf_out} with {len(pullf_files)} entries.")

    def run_wham(
        tpr_file_list="tpr-files.dat",
        pullf_file_list="pullf-files.dat",
        output_pmf="pmf.xvg",
        temperature=323
    ):
        # Convert paths to WSL
        cwd_wsl = Path.cwd().as_posix().replace("C:", "/mnt/c")
        tpr_file_list_wsl = f"{cwd_wsl}/{tpr_file_list}"
        pullf_file_list_wsl = f"{cwd_wsl}/{pullf_file_list}"
        output_pmf_wsl = f"{cwd_wsl}/{output_pmf}"
        
        bash_cmd = (
            f"source ~/martini-env/bin/activate && cd {cwd_wsl} && "
            f"gmx wham -it {tpr_file_list_wsl} -if {pullf_file_list_wsl} "
            f"-o {output_pmf_wsl} -temp {temperature}"
        )

        print("📊 Running WHAM to construct PMF inside WSL...")
        try:
            process = subprocess.Popen(
                ["wsl", "-e", "bash", "-c", bash_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end="")

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                print(f"✅ WHAM finished successfully. PMF saved to {output_pmf}")
            else:
                print(f"❌ WHAM failed with return code {return_code}")

        except Exception as e:
            print(f"❌ Exception while running WHAM: {e}")

    def analyze_pmf(
        pmf_filepath: str,
        simulation_dir: str = ".",
        reference_region: str = "end",
        n_points: int = 20
    ) -> float:
        """
        Analyzes GROMACS PMF .xvg file and computes G_bind, 
        while dynamically counting the number of umbrella sampling windows.

        Parameters
        ----------
        pmf_filepath : str
            Path to the PMF .xvg file (e.g., 'pmf.xvg').
        simulation_dir : str, default '.'
            Directory to search for umbrella window files (e.g., umbrella_01.gro).
        reference_region : str, default 'start'
            Region to define the unbound reference ('start' or 'end').
        n_points : int, default 20
            Number of points to average for the reference region.

        Returns
        -------
        delta_g : float
            Gibbs Free Energy of Binding (G_bind) in kJ/mol.
        """

        def count_umbrella_windows(directory: str) -> int:
            pattern = re.compile(r"umbrella_(\d+)\.gro$")
            count = 0
            for fname in os.listdir(directory):
                if pattern.match(fname):
                    count += 1
            return count

        def read_xvg(path: str) -> Tuple[list, list]:
            xs, ys = [], []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(("@", "#")) or not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        xs.append(float(parts[0]))
                        ys.append(float(parts[1]))
            return xs, ys

        def get_binding_energy(xs: list, ys: list, region: str, n: int) -> Tuple[float, float, float, float]:
            if len(ys) < n:
                raise ValueError("Not enough data points to compute reference energy.")
            if region == 'start':
                ref_energy = sum(ys[:n]) / n
            elif region == 'end':
                ref_energy = sum(ys[-n:]) / n
            else:
                raise ValueError("reference_region must be 'start' or 'end'")
            min_energy = min(ys)
            delta_g = ref_energy - min_energy
            min_index = ys.index(min_energy)
            return delta_g, ref_energy, min_energy, xs[min_index]

        num_windows = count_umbrella_windows(simulation_dir)
        print(f"🔍 Detected umbrella windows: {num_windows}")

        xs, ys = read_xvg(pmf_filepath)
        delta_g, ref, min_val, min_pos = get_binding_energy(xs, ys, region=reference_region, n=n_points)

        print(f"Gibbs Free Energy of Binding (G_bind): {delta_g:.3f} kJ/mol")
        print(f"Reference (unbound) energy: {ref:.3f} kJ/mol")
        print(f"Minimum (bound) energy: {min_val:.3f} kJ/mol at {min_pos:.3f} nm")

        with open("pmf_analysis.txt", "w") as f:
            f.write(f"Umbrella Windows: {num_windows}\n")
            f.write(f"Gibbs Free Energy of Binding (G_bind): {delta_g:.3f} kJ/mol\n")
            f.write(f"Reference (unbound) energy: {ref:.3f} kJ/mol\n")
            f.write(f"Minimum (bound) energy: {min_val:.3f} kJ/mol at {min_pos:.3f} nm\n")

        return delta_g

    def analyze_protein_po4_contacts(
        directory=".",
        protein_sel="protein",
        po4_sel="name PO4",
        cutoff_nm=1.0,
        gro_pattern="umbrella_*.gro",
        output_file="avg_contacts.txt"
    ):
        """
        Analyze PO4–protein contacts in GRO files using MDAnalysis and save average.

        Parameters:
        - directory (str or Path): Folder containing .gro files.
        - protein_sel (str): Atom selection for protein.
        - po4_sel (str): Atom selection for PO4 beads.
        - cutoff_nm (float): Contact cutoff in nanometers.
        - gro_pattern (str): Pattern to match .gro files.
        - output_file (str): File to save average contact value.

        Returns:
        - contact_counts (list of int)
        - avg_contacts (float)
        """
        directory = Path(directory)
        gro_files = sorted(directory.glob(gro_pattern))
        contact_counts = []

        cutoff_angstrom = cutoff_nm * 10.0

        for gro in gro_files:
            u = mda.Universe(str(gro))

            prot_atoms = u.select_atoms(protein_sel)
            po4_atoms = u.select_atoms(po4_sel)

            print(f"{gro.name} → protein atoms: {len(prot_atoms)}, PO4 atoms: {len(po4_atoms)}")

            if len(prot_atoms) == 0 or len(po4_atoms) == 0:
                print(f"⚠️  Skipping {gro.name} due to empty selection.")
                continue

            distances = distance_array(prot_atoms.positions, po4_atoms.positions)
            n_contacts = np.sum(distances < cutoff_angstrom)
            contact_counts.append(n_contacts)

            print(f"{gro.name:20s} contacts = {n_contacts:4d}")

        if contact_counts:
            avg_contacts = np.mean(contact_counts)
            print(f"\n📊 Average PO4–protein contacts across {len(contact_counts)} windows: {avg_contacts:.2f}")

            # Save to file
            out_path = Path(output_file)
            out_path.write_text(f"{avg_contacts:.2f}\n")
            print(f"💾 Saved average to {out_path.resolve()}")

            return contact_counts, avg_contacts
        else:
            print("❌ No contacts calculated. Check selection syntax or .gro content.")
            return [], 0.0

    def main():
        # Get current working directory for consistent paths
        cwd = Path.cwd()
        print(f"Working directory: {cwd}")
        
        # Step 1: Convert PDB to coarse-grained structure
        gro_file, top_file = run_martinize_and_convert_gro_wsl(pdb_file, output_prefix)
        
        # Step 2: Position protein for membrane binding
        reposition_and_rotate_protein(
            protein_gro_in=f"{output_prefix}_cg.gro", 
            protein_gro_out="asyn_positioned.gro", 
            bilayer_box=[25.0, 25.0, 20.0], 
            upper_leaflet_z=20.0, 
            translate_z=0.0
        )
        
        # Step 3: Create membrane system
        setup_membrane_system_wsl(
            protein_gro_in="asyn_positioned.gro", 
            output_gro="system_solvated.gro", 
            output_top="topol_updated.top"
        )
        
        # Step 4: Convert topology to Martini 3
        convert_top_to_martini3("topol_updated.top")
        
        # Step 5: Create index files
        make_index_file("system_solvated.gro", ndx_file="index.ndx")
        rename_ndx_groups("index.ndx", "system_solvated.gro", "index_named.ndx")
        add_upper_leaflet_group_clustered(
            gro_file="system_solvated.gro", 
            ndx_file="index_named.ndx", 
            group_name="Upper_PO4"
        )
        
        # Step 6: Upload files to HPC
        upload_files_to_carya([
            "system_solvated.gro", 
            "index_named.ndx", 
            "Protein_0.itp", 
            "topol_updated.top"
        ])
        
        # Step 7: Run simulation steps on HPC
        run_gromacs_minimization()
        run_nvt_batch()
        run_npt_batch()
        run_pulling_batch()
        
        # Step 8: Download results and extract snapshots
        download_from_carya(["pull.xtc", "pull.tpr", "npt.gro"])
        extract_snapshots()
        
        # Step 9: Generate and run umbrella sampling windows
        submit_windows_to_npt()
        submit_windows_to_umbrella()
        
        # Step 10: Download results using rsync
        current_dir = str(cwd).replace("\\", "/")  # Normalize path separators
        run_rsync([
            "-avz", 
            "--include", "umbrella_*_pullf.xvg", 
            "--exclude", "*", 
            "mtelikep@carya.rcdc.uh.edu:/project/zerze/mtelikep/SFProject/", 
            current_dir
        ])
        run_rsync([
            "-avz", 
            "--include", "umbrella_*.tpr", 
            "--exclude", "*", 
            "mtelikep@carya.rcdc.uh.edu:/project/zerze/mtelikep/SFProject/", 
            current_dir
        ])
        run_rsync([
            "-avz", 
            "--include", "umbrella_*.gro", 
            "--exclude", "*", 
            "mtelikep@carya.rcdc.uh.edu:/project/zerze/mtelikep/SFProject/", 
            current_dir
        ])
        
        # Step 11: Analyze results
        create_file_lists()
        run_wham()
        
        # Step 12: Analyze PMF - make n_points dynamic based on available data
        try:
            analyze_pmf("pmf.xvg", reference_region='end', n_points=1)
        except ValueError as e:
            print(f"⚠️ PMF analysis error: {e}")
            print("Trying with fewer points...")
            analyze_pmf("pmf.xvg", reference_region='end', n_points=1)
        
        # Step 13: Analyze protein-lipid contacts
        analyze_protein_po4_contacts(directory=".", output_file="avg_contacts.txt")
        
        print("✅ MD dynamics pipeline completed successfully!")
    
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mdynamics Pipeline")
    parser.add_argument("--pdb", required=True, help="Path to input PDB file")
    parser.add_argument("--output_prefix", required=True, help="Output prefix for generated files")
    args = parser.parse_args()
    
    run_mdynamics_pipeline(pdb_file=args.pdb, output_prefix=args.output_prefix)