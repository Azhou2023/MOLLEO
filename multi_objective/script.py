import os
import shutil
import sys
import time
import numpy as np
import yaml
from similarity_clustering import cluster
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from scipy.stats import ttest_ind
# import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import re
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from collections import Counter, defaultdict


def get_atom_distribution(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    total_atoms = len(atom_symbols)
    atom_counts = Counter(atom_symbols)

    distribution = {}
    for symbol, count in atom_counts.items():
        percentage = (count / total_atoms) * 100
        distribution[symbol] = round(percentage, 2)
        
    sorted_distribution = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    return sorted_distribution


def get_average_atom_distribution(smiles_list):
    summed_distributions = defaultdict(float)
    valid_molecule_count = 0

    for smiles in smiles_list:
        dist = get_atom_distribution(smiles)
        
        if dist is not None:
            valid_molecule_count += 1
            for atom, pct in dist.items():
                summed_distributions[atom] += pct

    average_distribution = {}
    for atom, total_pct in summed_distributions.items():
        average_pct = total_pct / valid_molecule_count
        average_distribution[atom] = round(average_pct, 2)
    sorted_avg_dist = dict(sorted(average_distribution.items(), key=lambda item: item[1], reverse=True))

    return sorted_avg_dist

def analyze_results(run_name, pop_size=120, limit=False, llm_only=True, eval_sim=False, num_seeds=3):
    print("RUN: " + run_name)
    
    all_llm_ligands = [[] for _ in range(num_seeds)]
    num_errors = 0
    curr_seed = 0
    seeds_passed = 0
    if llm_only:
        with open(f"/home/ubuntu/MOLLEO/multi_objective/logs/{run_name}.txt", 'r') as log_file:
            for line in log_file:
                if "seed" in line and line.split()[0]=="seed":
                    curr_seed = int(line.split()[1].strip())
                    seeds_passed += 1
                if "1000/1000" in line and seeds_passed == num_seeds:
                    break
                if "LLM-GENERATED:" in line:
                    ligand = line.split()[1].strip()
                    all_llm_ligands[curr_seed].append(ligand)
                if "NUM LLM ERRORS" in line:
                    num_errors += 1
    avg_top_10 = []
    avg_top_10_qed = 0
    avg_top_10_sa = 0
    avg_top_10_max_sim = 0
    better_than_threshold = 0
    
    avg_max_sim_filtered = 0
    unique_mean = []
    unique_filtered_mean = []
    
    num_filtered = 0
    filter_mean = 0
    
    pareto_size = 0
    pareto_mean = 0
    pareto_qed_mean = 0
    pareto_sa_mean = 0
    hypervolume = 0
    hv_ttest = []
    fmean_ttest = []
    
    top_auc_score = 0
    
    output_mols = []
    morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    affins = np.zeros((num_seeds, 1000-pop_size)) * np.nan
    
    all_ligands = []
    for seed, llm_ligands in enumerate(all_llm_ligands):
        cmet = []
        ligands = {}
        cache = {}
        top_auc = np.zeros((3, 1000-pop_size)) * np.nan
        top_affins = []
        top_qed = []
        top_sa = []
        filtered_ligands = []
        with open(f"/home/ubuntu/MOLLEO/multi_objective/results/{run_name}/seed_{seed}.yaml", 'r') as file:
            data = yaml.safe_load(file)
            sorted_data = sorted(data, key=lambda k: data[k][1])
            for idx, ligand in enumerate(sorted_data):
                if limit and idx > 255:
                    continue
                if idx < pop_size:
                    cmet.append(ligand)
                    cache[ligand] = float(data[ligand][0])
                    continue
                if (llm_only is False or ligand in llm_ligands) and float(data[ligand][0])<0:
                    affin = float(data[ligand][0])
                    qed = float(data[ligand][2])
                    sa = float(data[ligand][3])
                    # mw = Descriptors.MolWt(mol)
                    
                    if qed > 0.5 and sa < 3.0:
                        filtered_ligands.append(ligand)
                    
                    scaled_affin = max(-((affin+6)/6), 0)
                    scaled_qed = max((qed-0.5)*2, 0)
                    scaled_sa = max(((1-(sa-1)/9) - 0.5)*2, 0)
                    if len(top_affins) < 10:
                        top_affins.append(scaled_affin)
                        top_qed.append(scaled_qed)
                        top_sa.append(scaled_sa)
                    else:
                        if scaled_affin > min(top_affins):
                            top_affins[np.argmin(top_affins)] = scaled_affin
                        if scaled_qed > min(top_qed):
                            top_qed[np.argmin(top_qed)] = scaled_qed
                        if scaled_sa > min(top_sa):
                            top_sa[np.argmin(top_sa)] = scaled_sa
                    # best = min(best, float(data[ligand][0])) if best else float(data[ligand][0])        
                    top_auc[0][idx-pop_size] = np.nanmean(top_affins)
                    top_auc[1][idx-pop_size] = np.nanmean(top_qed)
                    top_auc[2][idx-pop_size] = np.nanmean(top_sa)

                    max_sim = 0
                    if eval_sim:
                        mol = Chem.MolFromSmiles(ligand)
                        fingerprint = morgan.GetFingerprint(mol)
                        
                        # scaf = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                        # fingerprint = AllChem.GetMorganFingerprintAsBitVect(scaf, radius=2, nBits=2048)

                        sim_ligand = ""
                        for cmet_ligand in cmet:
                            cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                            
                            cmet_fingerprint = morgan.GetFingerprint(cmet_mol)

                            # cmet_scaf = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(cmet_mol)
                            # cmet_fingerprint = AllChem.GetMorganFingerprintAsBitVect(cmet_scaf, radius=2, nBits=2048)
                            
                            similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                            
                            if similarity > max_sim:
                                max_sim = similarity
                                sim_ligand = cmet_ligand
                    ligands[ligand] = [affin, qed, sa, max_sim]
        top_auc_score += (np.trapz(top_auc[0]) + np.trapz(top_auc[1]) + np.trapz(top_auc[2]))
        print((np.trapz(top_auc[0]) + np.trapz(top_auc[1]) + np.trapz(top_auc[2])))
        
        # if "c-met" in run_name:
        #     with open(f"/home/ubuntu/MOLLEO/init_caches/c-met_{seed}.yaml", 'w') as file:
        #         yaml.dump(cache, file)
        # elif "brd4" in run_name:
        #     with open(f"/home/ubuntu/MOLLEO/init_caches/brd4_{seed}.yaml", 'w') as file:
        #         yaml.dump(cache, file)
        
        sorted_ligands = sorted(ligands, key=lambda k: ligands[k][0])
        sim_filtered_ligands = [ligand for ligand in sorted_ligands if ligands[ligand][3] < 0.5]
        c = cluster(sorted_ligands)
        c = sorted(c, key=lambda k: ligands[k][0])
        best_10_cluster = []
        qed = []
        sa = []
        sim = []
        threshold = -11
        num_better_than_threshold = 0
        print("Number of clusters: " + str(len(c)))
        for i in c[:10]:
            # print(i)
            best_10_cluster.append(ligands[i][0])
            qed.append(ligands[i][1])
            sa.append(ligands[i][2])
            sim.append(ligands[i][3])
            if ligands[i][0] < threshold:
                num_better_than_threshold += 1
        
        num_filtered += len(filtered_ligands)
        sorted_filtered = sorted(filtered_ligands, key=lambda k: ligands[k][0])
        best_10_filtered = []
        for i in sorted_filtered[:10]:
            best_10_filtered.append(ligands[i][0])
            # print(f"{i}: {str(ligands[i][0])}")
        c_filtered = cluster(sorted_filtered)
        c_filtered = sorted(c_filtered, key=lambda k: ligands[k][0])
        best_10_cluster_filtered = []
        for i in c_filtered[:10]:
            best_10_cluster_filtered.append(ligands[i][0])
            print(f"{i}: {str(ligands[i][0])} | {str(round(ligands[i][1], 2))} | {str(round(ligands[i][2], 2))}")
        
        filter_mean += np.mean(best_10_cluster_filtered)
        print(np.mean(best_10_cluster_filtered))
        fmean_ttest.append(np.mean(best_10_cluster_filtered))
        
        avg_top_10.append(np.mean(best_10_cluster))
        avg_top_10_qed += np.mean(qed)
        avg_top_10_sa += np.mean(sa)
        better_than_threshold += num_better_than_threshold
        avg_top_10_max_sim += np.mean(sim)
        
        if eval_sim:
            c_sim_filtered = cluster(sim_filtered_ligands)
            c_sim_filtered = sorted(c_sim_filtered, key=lambda k: ligands[k][0])
            for i in c_sim_filtered[:10]:
                print(i)
            avg_max_sim_filtered += np.mean([ligands[i][3] for i in sim_filtered_ligands])
            unique_mean.append(np.mean([ligands[i][0] for i in c_sim_filtered[:10]]))
            
            sim_filtered_filtered = [lig for lig in sim_filtered_ligands if (ligands[lig][1] > 0.5 and ligands[lig][2] < 3.0)]
            c_sim_filtered_filtered = cluster(sim_filtered_filtered)
            c_sim_filtered_filtered = sorted(c_sim_filtered_filtered, key=lambda k: ligands[k][0])
            unique_filtered_mean.append(np.mean([ligands[i][0] for i in c_sim_filtered_filtered[:10]]))

        # pareto analysis
        # ligands_list = list(ligands.keys())
        ligands_list = c
        score_list = []
        pareto_affins = []
        pareto_qed = []
        pareto_sa = []
        pareto_sim = []
        for ligand in ligands_list:
            pass_filter = True

            single_score = []
            
            single_score.append(1-(-ligands[ligand][0]/15))
            pareto_affins.append(ligands[ligand][0])
            
            single_score.append(1 - ligands[ligand][1])
            pareto_qed.append(ligands[ligand][1])
            
            single_score.append((ligands[ligand][2]-1)/9)
            pareto_sa.append(ligands[ligand][2])
            
            # if "bindingdb" in run_name or eval_sim: 
            #     mol = Chem.MolFromSmiles(ligand)
            #     fingerprint = morgan.GetFingerprint(mol)
            #     max_sim = 0
            #     for cmet_ligand in cmet:
            #         cmet_mol = Chem.MolFromSmiles(cmet_ligand)
            #         cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
            #         similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
            #         max_sim = max(max_sim, similarity)
            #     if max_sim > 0.3:
            #         pass_filter = False
                        
            if not eval_sim or max_sim < 0.3: 
                score_list.append(single_score)
            else:
                score_list.append([1.0, 1.0, 1.0])
        
        score_array = np.array(score_list)
        nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
        pareto_front = np.array(ligands_list)[nds]
        
        pareto_size += len(pareto_front)
        pareto_mean += np.mean(np.array(pareto_affins)[nds])
        pareto_qed_mean += np.mean(np.array(pareto_qed)[nds])
        pareto_sa_mean += np.mean(np.array(pareto_sa)[nds])
        hv = HV(ref_point=np.array([1.0, 1.0, 1.0]))
        vals = np.array(score_list)[nds]
        hv = hv(np.array(vals))
        
        hypervolume += hv
        hv_ttest.append(hv)
        print(hv)

        all_ligands.extend(sorted_ligands)
    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(avg_top_10)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(avg_top_10)))
    print("AVG QED (clustered): " + str(avg_top_10_qed / num_seeds))
    print("AVG SA (clustered): " + str(avg_top_10_sa / num_seeds))
    print("AVG MAX SIM TOP 10: " + str(avg_top_10_max_sim / num_seeds))
    print("NUM BETTER THAN THRESHOLD: " + str(better_than_threshold / num_seeds))
    print("UNIQUE MEAN: " + str(np.mean(unique_mean)))
    print("UNIQUE STDEV: " + str(np.std(unique_mean)))
    print("UNIQUE FILTERED MEAN: " + str(np.mean(unique_filtered_mean)))
    print("UNIQUE FILTERED STDEV: " + str(np.std(unique_filtered_mean)))
    print("AVERAGE UNIQUE MAX SIM: " + str(avg_max_sim_filtered / num_seeds))
    print()
    
    print("NUM FILTERED MOLECULES: " + str(num_filtered / num_seeds))
    print("MEAN FILTERED MOLECULES: " + str(filter_mean / num_seeds))
    print()
    
    print("PARETO FRONT SIZE: " + str(pareto_size / num_seeds))
    print("PARETO FRONT MEAN: " + str(pareto_mean / num_seeds))
    print("PARETO FRONT QED: " + str(pareto_qed_mean / num_seeds))
    print("PARETO FRONT SA: " + str(pareto_sa_mean / num_seeds))
    print("HYPERVOLUME: " + str(hypervolume / num_seeds))
    print()
    print("TOP-AUC: " + str(top_auc_score / num_seeds))
    print()
    return hv_ttest, fmean_ttest, all_ligands

# fig, axs = plt.subplots(1, 1)
# affins = analyze_results("GPT-oss_c-met_structure_info", limit=False, llm_only=True, eval_sim=False)
# axs.plot(np.nanmean(affins, axis=0))
# axs.set_yticks(np.arange(-15, 0, 1.0))
# interval = 1.96 * np.nanstd(affins, axis=0) / np.sqrt(affins.shape[0])
# axs.fill_between(np.arange(affins.shape[1]), np.nanmean(affins, axis=0) - interval, np.nanmean(affins, axis=0) + interval, alpha=0.2)
# plt.legend()
# plt.title("GA Trajectory: Filtered Mean")
# plt.ylabel("Boltz-2 Affinity (kcal/mol)")
# plt.xlabel("Steps")
# plt.ylim(-11, -8)
# plt.savefig('/home/ubuntu/MOLLEO/multi_objective/trajectory.png')

hv_ttest1, fmean_ttest1, ligands1 = analyze_results("GPT-oss_c-met_tools_exp_prob_60_35", pop_size=60, limit=False, llm_only=False, eval_sim=False, num_seeds=5)
# hv_ttest2, fmean_ttest2, ligands2 = analyze_results("GPT-oss_c-met_tools_molleo_ga_12_7", pop_size=12, limit=False, llm_only=False, eval_sim=False, num_seeds=5)
hv_ttest2, fmean_ttest2, ligands2 = analyze_results("GPT-oss_c-met_tools_exp_prob_10_60_35", pop_size=60, limit=False, llm_only=False, eval_sim=False, num_seeds=5)

bindingdb = []
with open("/home/ubuntu/LLaMA-Factory/bindingdb/cmet.txt", 'r') as file:
    for line in file:
        ligand = line.strip()
        bindingdb.append(ligand)
print(f"BindingDB atom distribution: {get_average_atom_distribution(bindingdb)}")

print(f"Run 1 atom distribution: {get_average_atom_distribution(ligands1)}")
print(f"Run 2 atom distribution: {get_average_atom_distribution(ligands2)}")

print(f"[Run 1] Hypervolume std: {str(np.std(hv_ttest1))} | Filtered mean std: {str(np.std(fmean_ttest1))}")
print(f"[Run 2] Hypervolume std: {str(np.std(hv_ttest2))} | Filtered mean std: {str(np.std(fmean_ttest2))}")

_, hv_p = ttest_ind(hv_ttest1, hv_ttest2, alternative="greater", equal_var=False)
_, fm_p = ttest_ind(fmean_ttest1, fmean_ttest2, alternative="less", equal_var=False)
print(f"Hypervolume p-value (run 1 > run 2): {str(hv_p)}")
print(f"Filtered mean p-value (run 1 < run 2): {str(fm_p)}")
sys.exit(0)

to_run = ["GPT-oss_c-met_tools_molleo_ga_60_35", "GPT-oss_c-met_molleo_scaled", "GPT-oss_c-met_molleo_12_7", "GPT-oss_c-met_tools_molleo_ga", "GPT-oss_c-met_tools_molleo_ga_12_7", "GPT-oss_c-met_tools_top_3_pareto_prob", "GPT-oss_c-met_tools_top_3_pareto_prob_12_7", "GPT-oss_c-met_tools_top_3_scalarized_prob", "GPT-oss_c-met_tools_crowding_12_7"]
hv_ranking = {}
fmean_ranking = {}

hv_ttest1, fmean_ttest1, ligands1 = analyze_results("GPT-oss_c-met_top_3_pareto_prob", pop_size=12, limit=False, llm_only=False, eval_sim=False, num_seeds=3)
hv_ranking["GPT-oss_c-met_top_3_pareto_prob"] = np.mean(hv_ttest1)
fmean_ranking["GPT-oss_c-met_top_3_pareto_prob"] = np.mean(fmean_ttest1)
for run in to_run:
    hv_ttest1, fmean_ttest1, ligands1 = analyze_results(run, pop_size=12, limit=False, llm_only=False, eval_sim=False, num_seeds=5)
    hv_ranking[run] = np.mean(hv_ttest1)
    fmean_ranking[run] = np.mean(fmean_ttest1)

sorted_hv = sorted(hv_ranking, key=hv_ranking.get, reverse=True)
sorted_fmean = sorted(fmean_ranking, key=fmean_ranking.get)

print("HYPERVOLUME:")
for hv in sorted_hv:
    print(f"{hv}: {str(hv_ranking[hv])}")
print()
print("FILTERED MEAN:")
for fmean in sorted_fmean:
    print(f"{fmean}: {str(fmean_ranking[fmean])}")

# create_yaml("GPT-4_c-met_zinc")
# set_similarity()

# home_path = "/home/ubuntu/MOLLEO/multi_objective/results"
# subdirectories = [f for f in os.listdir(home_path) if os.path.isdir(os.path.join(home_path, f))]
