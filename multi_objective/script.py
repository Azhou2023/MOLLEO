import os
import shutil
import sys
import numpy as np
import yaml
from similarity_clustering import cluster
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
import re
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def analyze_results(run_name, limit=False, llm_only=True, eval_sim=False, num_seeds=3):
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
    
    num_filtered = 0
    filter_mean = 0
    
    pareto_size = 0
    pareto_mean = 0
    pareto_qed_mean = 0
    pareto_sa_mean = 0
    hypervolume = 0
    t_test = []
    
    output_mols = []
    morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    affins = np.zeros((num_seeds, 880)) * np.nan
    top_filtered_affins = []
    for seed, llm_ligands in enumerate(all_llm_ligands):
        cmet = []
        ligands = {}
        cache = {}
        with open(f"/home/ubuntu/MOLLEO/multi_objective/results/results_{run_name}_{seed}.yaml", 'r') as file:
            data = yaml.safe_load(file)
            sorted_data = sorted(data, key=lambda k: data[k][1])
            for idx, ligand in enumerate(sorted_data):
                if limit and idx > 255:
                    continue
                if idx < 120:
                    cmet.append(ligand)
                    cache[ligand] = float(data[ligand][0])
                else:
                    if (llm_only is False or ligand in llm_ligands) and float(data[ligand][0])<0:
                        affin = float(data[ligand][0])
                        mol = Chem.MolFromSmiles(ligand)
                        qed = QED.qed(mol)
                        sa = sascorer.calculateScore(mol)
                        # mw = Descriptors.MolWt(mol)
                        if qed > 0.5 and sa < 3.0:
                            if len(top_filtered_affins) < 10:
                                top_filtered_affins.append(affin)
                            else:
                                if affin < top_filtered_affins[np.argmax(top_filtered_affins)]:
                                    top_filtered_affins[np.argmax(top_filtered_affins)] = affin
                        # best = min(best, float(data[ligand][0])) if best else float(data[ligand][0])
                        affins[seed][idx-120] = np.mean(top_filtered_affins)
                        max_sim = 0
                        if eval_sim:
                            fingerprint = morgan.GetFingerprint(mol)
                            
                            # scaf = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                            # fingerprint = AllChem.GetMorganFingerprintAsBitVect(scaf, radius=2, nBits=2048)

                            # sim_ligand = ""
                            for cmet_ligand in cmet:
                                cmet_mol = Chem.MolFromSmiles(cmet_ligand)
                                
                                cmet_fingerprint = morgan.GetFingerprint(cmet_mol)

                                # cmet_scaf = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(cmet_mol)
                                # cmet_fingerprint = AllChem.GetMorganFingerprintAsBitVect(cmet_scaf, radius=2, nBits=2048)
                                
                                similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
                                
                                if similarity > max_sim:
                                    max_sim = similarity
                                    # sim_ligand = cmet_ligand
                        ligands[ligand] = [affin, qed, sa, max_sim]
                        
        if "c-met" in run_name:
            with open(f"/home/ubuntu/MOLLEO/init_caches/c-met_{seed}.yaml", 'w') as file:
                yaml.dump(cache, file)
        elif "brd4" in run_name:
            with open(f"/home/ubuntu/MOLLEO/init_caches/brd4_{seed}.yaml", 'w') as file:
                yaml.dump(cache, file)
        sorted_ligands = sorted(ligands, key=lambda k: ligands[k][0])
        filtered_ligands = [ligand for ligand in sorted_ligands if ligands[ligand][3] < 0.5]
        
        c = cluster(sorted_ligands)
        c = sorted(c, key=lambda k: ligands[k][0])
        best_10_cluster = []
        qed = []
        sa = []
        sim = []
        threshold = -11
        num_better_than_threshold = 0
        for i in c[:10]:
            # print(i)
            best_10_cluster.append(ligands[i][0])
            qed.append(ligands[i][1])
            sa.append(ligands[i][2])
            sim.append(ligands[i][3])
            if ligands[i][0] < threshold:
                num_better_than_threshold += 1
            
        c_filtered = cluster(filtered_ligands)
        c_filtered = sorted(c_filtered, key=lambda k: ligands[k][0])
        # for i in c_filtered[:10]:
        #     print(i)
        
        avg_top_10.append(np.mean(best_10_cluster))
        avg_top_10_qed += np.mean(qed)
        avg_top_10_sa += np.mean(sa)
        better_than_threshold += num_better_than_threshold
        avg_top_10_max_sim += np.mean(sim)
        
        avg_max_sim_filtered += np.mean([ligands[i][3] for i in filtered_ligands])
        unique_mean.append(np.mean([ligands[i][0] for i in c_filtered[:10]]))
        # print("AVG TOP TEN: " + str(np.mean(best_10)))
        # print("AVG TOP TEN (CLUSTERED): " + str(np.mean(best_10_cluster)))
        # print("BEST: " + str(min(best_10_cluster)))
        # print("STDEV TOP 10 (CLUSTERED): " + str(np.std(best_10_cluster)))
        # print("BEST 10 LIGANDS (CLUSTERED):")
                
        # print("AVG QED (clustered): " + str(np.mean(qed)))
        # print("AVG SA (clustered): " + str(np.mean(sa)))
        # print("STDEV QED: " + str(np.std(qed)))
        # print("AVG MAX SIM: " + str(np.mean(sim)))
        # print("NUMBER OF LLM ERRORS: " + str(num_errors))
        # print("NUM BETTER THAN THRESHOLD: " + str(num_better_than_threshold))
        # print("UNIQUE GENERATION MEAN: " + str(np.mean(sorted(unique)[:10])))
        # print("UNIQUE GENERATION STD: " + str(np.std(sorted(unique)[:10])))
        
        # pareto analysis
        ligands_list = list(ligands.keys())
        score_list = []
        pareto_affins = []
        pareto_qed = []
        pareto_sa = []
        pareto_sim = []
        filtered_mols = []
        for ligand in ligands_list:
            pass_filter = True

            single_score = []
            
            single_score.append(1-(-ligands[ligand][0]/15))
            pareto_affins.append(ligands[ligand][0])
            
            single_score.append(1 - ligands[ligand][1])
            pareto_qed.append(ligands[ligand][1])
            if ligands[ligand][1] < 0.5:
                pass_filter = False
            
            single_score.append((ligands[ligand][2]-1)/9)
            pareto_sa.append(ligands[ligand][2])
            if ligands[ligand][2] > 3.0:
                pass_filter = False
            
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
                
            if pass_filter:
                filtered_mols.append(ligand)
                
        num_filtered += len(filtered_mols)
        # print("NUM FILTERED MOLECULES: " + str(len(filtered_mols)))
        # print("MEAN FILTERED MOLECULES: " + str(np.mean([ligands[k][0] for k in filtered_mols])))
        sorted_filtered = sorted(filtered_mols, key=lambda k: ligands[k][0])
        best_10_filtered = []
        for i in sorted_filtered[:10]:
            best_10_filtered.append(ligands[i][0])
        
        c_filtered = cluster(sorted_filtered)
        c_filtered = sorted(c_filtered, key=lambda k: ligands[k][0])
        best_10_cluster_filtered = []
        for i in c_filtered[:10]:
            best_10_cluster_filtered.append(ligands[i][0])
        
        filter_mean += np.mean(best_10_cluster_filtered)
        # t_test.append(np.mean(best_10_cluster_filtered))
        # print("MEAN TOP 10 FILTERED MOLECULES: " + str(np.mean(best_10_filtered)))
        # print("MEAN TOP 10 FILTERED MOLECULES (CLUSTERED): " + str(np.mean(best_10_cluster_filtered)))
        
        score_array = np.array(score_list)
        nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
        pareto_front = np.array(ligands_list)[nds]
        
        pareto_size += len(pareto_front)
        pareto_mean += np.mean(np.array(pareto_affins)[nds])
        pareto_qed_mean += np.mean(np.array(pareto_qed)[nds])
        pareto_sa_mean += np.mean(np.array(pareto_sa)[nds])
        # print("PARETO FRONT SIZE: " + str(len(pareto_front)))
        # print("PARETO FRONT MEAN: " + str(np.mean(pareto_affins)))
        # print("PARETO FRONT QED: " + str(np.mean(pareto_qed)))
        # print("PARETO FRONT SA: " + str(np.mean(pareto_sa)))
        hv = HV(ref_point=np.array([1.0, 1.0, 1.0]))
        vals = np.array(score_list)[nds]
        hv = hv(np.array(vals))
        
        hypervolume += hv
        t_test.append(hv)
        print(hv)
        # print("HYPERVOLUME: " + str(hv))
        # get_all_ligands()
        
        # output_mols.append({key: ligands[key]} for key in c[:10])
        # m_weights = [Descriptors.MolWt(Chem.MolFromSmiles(ligand)) for ligand in sorted_data]
        # print(m_weights)
        # running_avg = []
        # cumsum = 0.0
        # for i in range(len(m_weights)):
        #     if i == 0:
        #         running_avg.append(None)  # no values before index 0
        #     else:
        #         cumsum += m_weights[i - 1]
        #         running_avg.append(cumsum / i)

        # x = list(range(len(m_weights)))

        # plt.figure()
        # plt.plot(x, running_avg, marker='o')
        # plt.xlabel("Index")
        # plt.ylabel("Average of values before index")
        # plt.title("Running Average (excluding current value)")
        # plt.show()

    print("AVG TOP TEN (CLUSTERED): " + str(np.mean(avg_top_10)))
    print("STDEV TOP 10 (CLUSTERED): " + str(np.std(avg_top_10)))
    print("AVG QED (clustered): " + str(avg_top_10_qed / num_seeds))
    print("AVG SA (clustered): " + str(avg_top_10_sa / num_seeds))
    print("AVG MAX SIM TOP 10: " + str(avg_top_10_max_sim / num_seeds))
    print("NUM BETTER THAN THRESHOLD: " + str(better_than_threshold / num_seeds))
    print("UNIQUE MEAN: " + str(np.mean(unique_mean)))
    print("UNIQUE STDEV: " + str(np.std(unique_mean)))
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
    print()
    return t_test

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

# values1 = analyze_results("GPT-oss_c-met_affin_prompt", limit=False, llm_only=False, eval_sim=False, num_seeds=5)
# values2 = analyze_results("GPT-oss_c-met_structure_info", limit=False, llm_only=False, eval_sim=False, num_seeds=5)

# _, p = ttest_ind(values1, values2, alternative="greater", equal_var=False)
# print(p)
# create_yaml("GPT-4_c-met_zinc")
# set_similarity()

home_path = "/home/ubuntu/MOLLEO/multi_objective/results"
subdirectories = [f for f in os.listdir(home_path) if os.path.isdir(os.path.join(home_path, f))]

for subdir in subdirectories:
    for filename in os.listdir(os.path.join(home_path, subdir)):
        new_buffer = {}
        with open(os.path.join(home_path, subdir, filename), 'r') as file:
            data = yaml.safe_load(file)
            for smiles in data:
                if len(data[smiles]==4):
                    break
                mol = Chem.MolFromSmiles(smiles)
                qed = QED.qed(mol)
                sa = sascorer.calculateScore(mol)
                
                new_buffer[smiles] = data[smiles, data[smiles][1], qed, sa]
        with open(os.path.join(home_path, subdir, filename), 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)