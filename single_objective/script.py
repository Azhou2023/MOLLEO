import os
import numpy as np
import yaml
from similarity_clustering import cluster
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import DataStructs
from scipy.stats import ttest_ind

def get_all_ligands():
    with open("single_objective/results_GPT-4_cmet_boltz.yaml", 'r') as file:
        current_ligand = ""
        with open("brd4.txt", 'w') as outfile:
            num_ligands = 0
            for line in file:
                if line[0]==":":
                    break
                if line[0]!="-":
                    current_ligand = line[:-2]
                    current_ligand = str.replace(current_ligand, ":", '')
                if line[0]=="-":
                    parts = line.split()
                    if '.' in parts[1]:
                        affin = -float(parts[1])
                        if affin==0: continue
                        outfile.write(current_ligand+"\t"+str(affin)+"\n")
                        num_ligands+=1

def set_similarity():
    ligands = []
    with open("/home/ubuntu/MOLLEO/single_objective/data/RAG_sample.txt", 'r') as file:
        for line in file:
            ligand = line.strip().replace('"', '')
            ligands.append(ligand)
    sample = np.random.choice(ligands, size=120, replace=False)
    mols = [Chem.MolFromSmiles(s) for s in sample]
    morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    fps = [morgan.GetFingerprint(m) for m in mols]
    sims = []
    n = len(fps)
    qeds = []
    for i in range(n):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        qeds.append(QED.qed(mols[i]))
    print("Mean similarity:", np.mean(sims))
    print("Mean QED:", np.mean(qeds))
    
def quoted_presenter(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    
def create_yaml(run_name):
    mol_buffer = {}
    num_mols = 0
    curr_ligand = ""
    mark = False
    with open(f"single_objective/logs/{run_name}.txt") as file:
        for line in file:
            if len(line.split()) > 1 and line.split()[0] == "1000/1000":
                break
            if "Boltz running on GPU" not in line and line!="\n" and not mark:
                curr_ligand = line.strip()
            if "Boltz running on GPU" in line:
                mark = True
                continue
            if mark:
                try:
                    fitness = float(line.strip())
                except:
                    fitness = -0.0
                num_mols += 1
                mol_buffer[curr_ligand] = [fitness, num_mols]
                mark = False
    mol_buffer = dict(sorted(mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    yaml.add_representer(str, quoted_presenter)
    with open(f"single_objective/results/{run_name}.yaml", 'w') as f:
        yaml.dump(mol_buffer, f, sort_keys=False)
        
def analyze_results(run_name, bindingdb=True, llm_only=True):
    print("RUN: " + run_name)
    
    llm_ligands = []
    num_errors = 0
    if llm_only:
        with open(f"single_objective/logs/{run_name}.txt", 'r') as log_file:
            for line in log_file:
                if "LLM-GENERATED:" in line:
                    ligand = line.split()[1].strip()
                    llm_ligands.append(ligand)
                if "NUM LLM ERRORS" in line:
                    num_errors += 1
   
    cmet = []
    ligands = {}
    with open(f"single_objective/results/{run_name}.yaml", 'r') as file:
        data = yaml.safe_load(file)
        for ligand, values in data.items():
            if int(values[1]) <= 120:
                cmet.append(ligand)
            else:
                if llm_only is False or ligand in llm_ligands:
                    ligands[ligand] = -float(values[0])
                
    sorted_ligands = sorted(ligands, key=ligands.get)
    print(len(cmet))
    print(len(sorted_ligands))
    best_10 = []
    for i in sorted_ligands[:10]:
        best_10.append(ligands[i])
    
    c = cluster(sorted_ligands)
    c = sorted(c, key=ligands.get)
    best_10_cluster = []
    for i in c[:10]:
        best_10_cluster.append(ligands[i])
    print("AVG TOP TEN: " + str(np.mean(best_10)))
    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(best_10_cluster)))
    print("BEST: " + str(min(best_10_cluster)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(best_10_cluster)))
    print("BEST 10 LIGANDS (CLUSTERED):")
    qed = []
    sim = []
    num_better_than_threshold = 0
    threshold = -11
    for idx, ligand in enumerate(c):
        if ligands[ligand] < threshold:
            num_better_than_threshold += 1
        if idx < 0:
            mol = Chem.MolFromSmiles(ligand)
            qed_score = QED.qed(mol)
            qed.append(qed_score)
            
            morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
            fingerprint = morgan.GetFingerprint(mol)
            max_sim = 0
            for cmet_ligand in cmet:
                cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
                similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                max_sim = max(similarity, max_sim)
                if similarity == 1.0: print(cmet_ligand)
            sim.append(max_sim)
            print(ligand + " " + str(ligands[ligand]))
    print("AVG QED (clustered): " + str(np.mean(qed)))
    print("STDEV QED: " + str(np.std(qed)))
    print("AVG MAX SIM: " + str(np.mean(sim)))
    print("NUMBER OF LLM ERRORS: " + str(num_errors))
    print("NUM BETTER THAN THRESHOLD: " + str(num_better_than_threshold))
    return best_10_cluster
# get_all_ligands()
# values1 = analyze_results("custom_c-met_no_summary_sft", bindingdb=True, llm_only=True)
# values2 = analyze_results("custom_c-met_untuned_llama", bindingdb=True, llm_only=True)
# _, p = ttest_ind(values1, values2, alternative="less", equal_var=False)
# print(p)
# create_yaml("GPT-4_c-met_zinc")
# set_similarity()

runs = [filename.replace(".yaml", "") for filename in os.listdir("single_objective/results") if "custom_c-met" in filename]
print(runs)
results = {}
for run in runs:
    results[run] = np.mean(analyze_results(run, bindingdb=True, llm_only=True))
sorted_results = sorted(results, key=results.get)
for result in sorted_results:
    print(f"{result}: {str(results[result])}")