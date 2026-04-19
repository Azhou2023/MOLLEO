import os
import requests
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import tdc
from tdc.generation import MolGen
from main.utils.chem import *
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from .boltz import calculate_boltz
from collections import defaultdict
from torch import multiprocessing as mp
from queue import Empty
from kubernetes import client, config
from .docking import calculate_docking

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

num_gpus = torch.cuda.device_count()
API_URL = "https://andrew-boltz-api.nrp-nautilus.io/predict_affinity"
use_nautilus = True
use_local = False

class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

def tuple_to_score(input_tuple, oracle, *args):
    for idx, element in enumerate(input_tuple):
        if args[idx][0] == "c-met" or args[idx][0] == "brd4":
            affin = -(element/13) if oracle == "boltz" else -(element/15)
        elif args[idx][0] == "qed":
            qed = (1 - element)
        elif args[idx][0] == "sa":
            sa = (1 - element)
        else:
            raise Exception("Invalid evaluator")
        
    bias = {"qed": 1, "sa": 1, "affin": 1}
    weights = {"qed": 1, "sa": 1, "affin": 1}

    qed = qed * bias["qed"] * weights["qed"]
    sa = sa * bias["sa"] * weights["sa"]
    affin = affin * bias["affin"] * weights["affin"]
    
    return affin + qed + sa

def get_ready_pod_count(namespace, deployment_name):

    try:
        config.load_kube_config()
    except config.ConfigException:
        config.load_incluster_config()

    api_instance = client.AppsV1Api()

    try:
        deployment = api_instance.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )

        ready_count = deployment.status.ready_replicas or 0
        print(f"Deployment: {deployment_name}")
        print(f"Total Desired: {deployment.status.replicas}")
        print(f"Ready & Running: {ready_count}")
        
        return ready_count

    except Exception as e:
        if e.status == 404:
            print(f"Error: Deployment '{deployment_name}' not found in namespace '{namespace}'.")
        else:
            print(f"API Error: {e}")
        return -1

def calculate_boltz_nautilus(protein_name, ligand_smiles, idx):
    print(f"\n[Worker {str(idx)}] Sending job for {ligand_smiles}...", flush=True)
    
    query_params = {
        "protein_name": protein_name,
        "ligand": ligand_smiles
    }
    worker_lifetime = time.time()
    num_errors = 0
    while True:
        try:
            before = time.time()
            response = requests.post(API_URL, params=query_params, headers={'Connection': 'close'})
            after = time.time()
            print(response)
            if response.status_code == 200:
                result = response.json()
                print(f"\n[Worker {str(idx)}] Success")
                print(result)
                print(f"[Worker {str(idx)}]Took: {str(after-before)} seconds", flush=True)
                time.sleep(0.1)
                return result["affinity"]
            elif response.status_code == 429 or response.status_code == 503:
                sleep_time = random.uniform(0.5, 2.0)
                print(f"[Worker {str(idx)}] Busy: Retrying", flush=True)
                current_lifetime = time.time()
                if worker_lifetime - current_lifetime > 3600:
                    print(f"[Worker {str(idx)}] Worker has been stalled for {str(worker_lifetime-current_lifetime)} seconds. Exiting.", flush=True)
                    return 0
                time.sleep(sleep_time)
            else:
                print(f"[Worker {str(idx)}] Error {response.status_code}: {response.text}", flush=True)
                num_errors += 1
                if num_errors >= 100:
                    return 0
                time.sleep(10)
        except Exception as e:
            print(f"\n[Worker {str(idx)}] Connection failed: {str(e)}", flush=True)        
            time.sleep(3)
            return 0

def gpu_worker(gpu_id, task_q, result_q, max_evaluator, min_evaluator, affin_cache, oracle):
    while True:
        try:
            idx, smi, val = task_q.get(timeout=3)
            try:
                if smi is None:
                    zeros_max = [0] * len(max_evaluator)
                    zeros_min = [0] * len(min_evaluator)
                    result_q.put((idx, smi, zeros_max + zeros_min, zeros_max + zeros_min))  
                else:
                    single_score = []
                    affin_scores = []
                    for eva_tuple in max_evaluator:
                        eva_name = eva_tuple[0] 
                        eva = eva_tuple[1]
                        if eva_name == 'qed':
                            mol = Chem.MolFromSmiles(smi)
                            single_score.append(1 - eva(mol))
                        elif eva_name == "c-met" or eva_name == "brd4":
                            if val is not None or smi in affin_cache[eva_name]:
                                affin = affin_cache[eva_name][smi]
                            else:
                                if oracle == "boltz":
                                    if gpu_id >= num_gpus or not use_local:
                                        affin = calculate_boltz_nautilus(eva_name, smi, gpu_id)
                                    else:
                                        affin = calculate_boltz(eva_name, smi, gpu_id)
                                elif oracle == "docking":
                                    affin = calculate_docking(eva_name, smi)
                            affin_scores.append(affin)
                            single_score.append(-affin)
                        else:
                            single_score.append(1 - eva(smi))
                    for eva_tuple in min_evaluator:
                        eva_name = eva_tuple[0]
                        eva = eva_tuple[1]
                        if eva_name == 'sa':
                            mol = Chem.MolFromSmiles(smi)
                            single_score.append((eva(mol) - 1)/9)
                        elif eva_name == "c-met" or eva_name == "brd4":
                            if val is not None or smi in affin_cache[eva_name]:
                                affin = affin_cache[eva_name][smi]
                            else:
                                if oracle == "boltz":
                                    if gpu_id >= num_gpus or not use_local:
                                        affin = calculate_boltz_nautilus(eva_name, smi, gpu_id)
                                    else:
                                        affin = calculate_boltz(eva_name, smi, gpu_id)
                                elif oracle == "docking":
                                    affin = calculate_docking(eva_name, smi)
                            affin_scores.append(affin)
                            single_score.append(affin)
                    result_q.put((idx, smi, single_score, affin_scores))
                    if val is None: print(f"GPU {gpu_id} produced result: {str((smi, single_score, affin_scores))}\n", flush=True)
            except Exception as e:
                print(e)
                sys.exit()
        except Exception as e:
            print(e) 
            sys.exit()
            break
    
class Oracle:
    def __init__(self, args=None, config=None):
        self.name = None
        self.max_obj = args.max_obj
        self.min_obj = args.min_obj
        self.max_evaluator = None
        self.min_evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = {}
        self.storing_buffer = {}
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0
        self.config = config
        self.affin_cache = defaultdict(dict)
        self.affin_oracle = "boltz"
        self.starting_population = []

    @property
    def budget(self):
        return self.max_oracle_calls
    
    def parallel_oracle(self, inputs):
        print(inputs)
        print("Number of GPUs available: " + str(num_gpus))
        assert num_gpus > 0, "No GPUs available"

        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()

        # enqueue tasks
        num_added = 0
        for i, x in enumerate(inputs):
            if len(self.mol_buffer) + num_added >= self.max_oracle_calls:
                task_q.put((i, None, None))
                continue
            if x in self.mol_buffer:
                task_q.put((i, x, self.mol_buffer[x][0]))
            else:
                task_q.put((i, x, None))
                num_added += 1
        
        if self.affin_oracle == "boltz":
            num_nautilus = get_ready_pod_count("spatiotemporal-decision-making", "andrew-boltz-api")
            if use_nautilus and num_nautilus == -1:
                num_nautilus = 20
                
            num_workers = 0
            num_workers += num_gpus if use_local else 0
            num_workers += num_nautilus if use_nautilus else 0
            
            num_workers = 9
            print(f"{str(num_workers)} workers available ({str(num_gpus)} local, {str(num_nautilus)} on Nautilus)")
        elif self.affin_oracle == "docking":
            num_workers = 10
            
        if num_workers == 0:
            print("No workers available!")
            sys.exit(1)
            
        procs = []
        for gpu_id in range(min(num_workers, task_q.qsize())):
            p = ctx.Process(target=gpu_worker, args=(gpu_id, task_q, result_q, self.max_evaluator, self.min_evaluator, self.affin_cache, self.affin_oracle))
            p.start()
            procs.append(p)
            time.sleep(0.1)

        # collect results
        results = [None] * len(inputs)
        buffer_length = len(self.mol_buffer)
        num_added = 0
        for i in range(len(inputs)):
            idx, x, single_score, affin_scores = result_q.get()
            
            # results array
            results[idx] = single_score
            
            # boltz cache
            affin_idx = 0
            for eva_tuple in self.max_evaluator:
                if eva_tuple[0] == "c-met" or eva_tuple[0] == "brd4":
                    self.affin_cache[eva_tuple[0]][x] = affin_scores[affin_idx]
                    affin_idx += 1
            for eva_tuple in self.min_evaluator:
                if eva_tuple[0] == "c-met" or eva_tuple[0] == "brd4":
                    self.affin_cache[eva_tuple[0]][x] = affin_scores[affin_idx]
                    affin_idx += 1
            
            # mol buffer
            if x not in self.mol_buffer and any(single_score): 
                self.mol_buffer[x] = [tuple_to_score(single_score, self.affin_oracle, *self.max_evaluator, *self.min_evaluator), buffer_length+num_added+1]
                num_added += 1
                
        for p in procs:
            p.join()
        print("RESULTS: " + str(results))
        print(self.mol_buffer)
        return results
    
    def calculate_similarity(self, smiles):
        morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = morgan.GetFingerprint(mol)
        max_sim = 0
        for starting_ligand in self.starting_population:
            cmet_mol = Chem.MolFromSmiles(starting_ligand)
            cmet_fingerprint = morgan.GetFingerprint(cmet_mol)
            similarity = DataStructs.TanimotoSimilarity(fingerprint, cmet_fingerprint)
            max_sim = max(max_sim, similarity)
        return max_sim

    def assign_evaluator(self, args):

        self.max_evaluator = []
        self.min_evaluator = []
        for idx in range(len(self.max_obj)):
            if self.max_obj[idx] == "qed":
                self.max_evaluator.append(("qed", QED.qed))
            elif self.max_obj[idx] == "c-met":
                self.max_evaluator.append(("c-met", calculate_boltz))
            elif self.max_obj[idx] == "brd4":
                self.max_evaluator.append(("brd4", calculate_boltz))
            # eva = tdc.Oracle(name = self.max_obj[idx])
            # self.max_evaluator.append(eva)
        
        for idx in range(len(self.min_obj)):
            if self.min_obj[idx] == "sa":
                self.min_evaluator.append(("sa", sascorer.calculateScore))
            elif self.min_obj[idx] == "sim":
                self.min_evaluator.append(("sim", self.calculate_similarity))
            elif self.min_obj[idx] == "c-met":
                self.min_evaluator.append(("c-met", calculate_boltz))
            elif self.min_obj[idx] == "brd4":
                self.min_evaluator.append(("brd4", calculate_boltz))
            # eva = tdc.Oracle(name = self.min_obj[idx])
            # self.min_evaluator.append(eva)

    

    def evaluate(self, smi):
        print('\n'+smi)
        score = 0
        results = {}
        for eva_tuple in self.max_evaluator:
            eva_name = eva_tuple[0] 
            eva = eva_tuple[1]
            if eva_name == 'qed':
                mol = Chem.MolFromSmiles(smi)
                evaluation = eva(mol)
                score = score + evaluation
            elif eva_name == "c-met" or eva_name == "brd4":
                scale = 1
                if smi in self.affin_cache[eva_name]:
                    evaluation = self.affin_cache[eva_name][smi]
                else:
                    evaluation = eva(eva_name, smi)
                self.affin_cache[eva_name][smi] = evaluation
                score = score + scale * evaluation
            else:
                evaluation = eva(smi)
                score = score + evaluation
            results[eva_name] = evaluation
        for eva_tuple in self.min_evaluator:
            eva_name = eva_tuple[0]
            eva = eva_tuple[1]
            if eva_name == 'sa':
                mol = Chem.MolFromSmiles(smi)
                evaluation = eva(mol)
                score = score + (1 - ((evaluation - 1)/9))
            elif eva_name == "c-met" or eva_name == "brd4":
                scale = 1
                if smi in self.affin_cache[eva_name]:
                    evaluation = self.affin_cache[eva_name][smi]
                else:
                    evaluation = eva(eva_name, smi)
                self.affin_cache[eva_name][smi] = evaluation
                score = score + -scale * evaluation
            elif eva_name == "sim":
                if smi in self.starting_population:
                    evaluation = 0.5
                    score = score + evaluation
                else:
                    evaluation = eva(smi)
                    score = score + (1 - evaluation)
            results[eva_name] = evaluation
        for name in results:
            print(f"{name}: {str(results[name])}")
        print(f"Score: " + str(score))
        # print(self.affin_cache)
        return score

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def clean_buffer(self):
        self.storing_buffer = self.storing_buffer | self.mol_buffer
        self.mol_buffer = {}

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'seed_' + suffix + ".yaml")

        self.sort_buffer()
        new_buffer = {}
        main_target = ""
        for target in list(self.affin_cache.keys()):
            if target in self.min_obj:
                main_target = target
                break
        print("Main target: " + main_target)
        for smiles in self.mol_buffer:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
            new_buffer[smiles] = [self.affin_cache[main_target][smiles], self.mol_buffer[smiles][1], qed, sa]
        with open(output_file_path, 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m, canonical=True) for m in mols]
                n_calls = len(self.mol_buffer)
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        # try:
        print({
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100, 
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle": n_calls,
        })



    def __len__(self):
        return len(self.mol_buffer) 

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            # smi = Chem.MolToSmiles(mol, canonical=True)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluate(smi)), len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_tuples = self.parallel_oracle(smiles_lst)
            score_list = [tuple_to_score(score_tuple, self.affin_oracle, *self.max_evaluator, *self.min_evaluator) for score_tuple in score_tuples]
            print("Current buffer: " + str(self.mol_buffer)) 
            self.sort_buffer()
            self.log_intermediate()
            self.last_log = len(self.mol_buffer)
            self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    def crowding_distance(self, scores, front):
        distances = [0.0] * len(front)
        num_obj = len(scores[0])

        for i in range(num_obj):
            sorted_idx = sorted(range(len(front)), key=lambda x: scores[front[x]][i])
            min_val = scores[front[sorted_idx[0]]][i]
            max_val = scores[front[sorted_idx[-1]]][i]
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            for j in range(1, len(front) - 1):
                prev_val = scores[front[sorted_idx[j - 1]]][i]
                next_val = scores[front[sorted_idx[j + 1]]][i]
                if max_val - min_val == 0:
                    d = 0.0
                else:
                    d = (next_val - prev_val) / (max_val - min_val)
                distances[sorted_idx[j]] += d
        return distances
    

    def select_pareto_front(self, smiles_lst):
        if type(smiles_lst) == list:
            score_list = self.parallel_oracle(smiles_lst)
            score_array = np.array(score_list)
            n = 1
            
            # original
            nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
            print("Length of non-dominated front: " + str(len(nds)))
            return np.array(smiles_lst)[nds]
        
            # modified
            nds = NonDominatedSorting().do(score_array, only_non_dominated_front=False)[:n]
            print(nds)
            pareto_front_smiles = []
            num_appended = 0
            
            for i in range(len(nds)):            
                if num_appended >= self.config["population_size"]:
                    break
                print(len(nds[i]))
                if len(nds[i]) + num_appended <= self.config["population_size"]:
                    pareto_front_smiles.extend(list(np.array(smiles_lst)[nds[i]]))
                    num_appended += len(nds[i])
                else:
                    # crowding distance
                    print("Population size exceeded, employing crowding distance")
                    try:
                        cd = self.crowding_distance(score_list, nds[i])
                        front = list(np.array(smiles_lst)[nds[i]])
                        sorted_front = sorted(front, key=lambda x: cd[front.index(x)], reverse=True)
                        pareto_front_smiles.extend(sorted_front[:(self.config["population_size"]-num_appended)])
                    except Exception as e:
                        print(e)
                        pareto_front_smiles.extend(list(np.array(smiles_lst)[nds[i]])[:(self.config["population_size"]-num_appended)])
                    num_appended += len(nds[i])
            print("Pareto front length: " + str(len(pareto_front_smiles)))
            print("Non dominated front: " + str(list(np.array(smiles_lst)[nds[0]])))
            return pareto_front_smiles
        else:
            print('Smiles should be in the list format.')

    @property
    def finish(self):
        print("Length of buffer: " + str(len(self.mol_buffer)))
        print("Max oracle calls: " + str(self.max_oracle_calls))
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None, config=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.config = config
        self.oracle = Oracle(args=self.args, config=self.config)
        self.all_smiles = []
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:  
            if args.starting == "zinc":      
                data = MolGen(name = 'ZINC')
                self.all_smiles = data.get_data()['smiles'].tolist()
            else:
                with open(f"/home/ubuntu/MOLLEO/multi_objective/data/{args.min_obj[0]}.txt", "r") as file:
                    for line in file:
                        ligand = line.strip()
                        self.all_smiles.append(ligand)
            
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)
            
    def sanitize(self, smiles_list):
        new_smiles_list = []
        smiles_set = set()
        for smiles in smiles_list:
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        continue
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_smiles_list.append(smiles)
                except ValueError:
                    print('bad smiles')
        return new_smiles_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        

        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'seed_' + suffix + '.yaml')

        self.sort_buffer()
        new_buffer = {}
        main_target = ""
        for target in list(self.oracle.affin_cache.keys()):
            if target in self.oracle.min_obj:
                main_target = target
                break
            
        for smiles in self.oracle.mol_buffer:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
            sa = sascorer.calculateScore(mol)
            new_buffer[smiles] = [self.oracle.affin_cache[main_target][smiles], self.oracle.mol_buffer[smiles][1], qed, sa]
        with open(output_file_path, 'w') as f:
            yaml.dump(new_buffer, f, sort_keys=False)
    

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
            
    def optimize(self, seed=0, project="test"):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed 
        self.oracle.task_label = str(seed)
        self.oracle.config = self.config
        self.oracle.affin_oracle = self.args.oracle
        # initial affin values
        if self.seed <= 4:
            for eva in self.args.min_obj:
                if eva == "c-met" or eva == "brd4":
                    with open(f"/home/ubuntu/MOLLEO/{self.args.oracle}_caches/{eva}_{str(self.seed)}.yaml", 'r') as file:
                        self.oracle.affin_cache[eva] = yaml.safe_load(file)
                        print(self.oracle.affin_cache)
            for eva in self.args.max_obj:
                if eva == "c-met" or eva == "brd4":
                    with open(f"/home/ubuntu/MOLLEO/{self.args.oracle}_caches/{eva}_{str(self.seed)}.yaml", 'r') as file:
                        self.oracle.affin_cache[eva] = yaml.safe_load(file)
                        print(self.oracle.affin_cache)
                
        self._optimize(self.config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.oracle.task_label)
        self.reset()

