import os
from secrets import token_hex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

def calculate_docking(protein_name, smiles, autodock='~/AutoDock-GPU/bin/autodock_gpu_128wi', protein_file='/home/ubuntu/BAT.py/BAT-brd4/docking_files/LMCSS-5uf0_5uez_docked.maps.fld', num_devices=torch.cuda.device_count(), starting_device=0):
    working_directory = os.getcwd()
    run_id = token_hex(16)
    os.chdir("/home/andrew/docking")
    ligands_folder = run_id+"/inputs"
    outs_folder = run_id+"/outputs"
    if os.path.exists(ligands_folder):
        shutil.rmtree(ligands_folder)
    if os.path.exists(outs_folder):
        shutil.rmtree(outs_folder)
    os.makedirs(ligands_folder)
    os.makedirs(outs_folder)
    for device in range(starting_device, starting_device + num_devices):
        os.makedirs(f'{ligands_folder}/{device}')
        
    device = starting_device
    smiles = [smiles]*64
    print(protein_name)
    if protein_name == "c-met":
        protein_file = "/home/andrew/BAT-cmet-updated/docking_files/receptor.maps.fld"
    elif protein_name == "brd4":
        protein_file = "/home/andrew/BAT-cmet-updated/docking_files_brd4/LMCSS-5uf0_5uez_docked.maps.fld"
    else:
        raise Exception("Unknown protein")
    
    for i, smile in enumerate(smiles):
        subprocess.Popen(f'obabel -:"{smile}" -O {ligands_folder}/{device}/ligand{i}HASH{hash(smile)}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
        device += 1
        if device == starting_device + num_devices:
            device = starting_device
    while True:
        total = 0
        for device in range(starting_device, starting_device + num_devices):
            total += len(os.listdir(f'{ligands_folder}/{device}'))
        if total == len(smiles):
            break
    time.sleep(0.1)
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -L {ligands_folder}/0/ligand0.pdbqt -N {outs_folder}/ligand0', shell=True)
    else:
        ps = []
        for device in range(starting_device, starting_device + num_devices):
            ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -B {ligands_folder}/{device}/ligand*.pdbqt -N {outs_folder}/ -D {device + 1}', shell=True, stdout=subprocess.DEVNULL))
        stop = False
        while not stop: 
            stop = True
            for p in ps:
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in os.listdir(outs_folder):
        if file.endswith('.dlg'):
            content = open(f'{outs_folder}/{file}').read()
            if '0.000   0.000   0.000  0.00  0.00' not in content:
                try:
                    affins[int(file.split('ligand')[1].split('HASH')[0])] = float([line for line in content.split('\n') if 'RANKING' in line][0].split()[3])
                except:
                    pass
    shutil.rmtree(run_id)
    os.chdir(working_directory)

    return min(affins)
