from __future__ import print_function

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time
from typing import List
import math

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main.molleo_multi_pareto.crossover as co, main.molleo_multi_pareto.mutate as mu
from main.pareto_optimizer import BaseOptimizer, calculate_boltz_nautilus, POSSIBLE_TARGETS

#from main.graph_ga.mol_lm import MolCLIP
from main.molleo_multi_pareto.biot5 import BioT5
from main.molleo_multi_pareto.GPT4 import GPT4
from main.molleo_multi_pareto.GPToss import GPToss
from utils import get_fp_scores
from network import create_and_train_network, obtain_model_pred

MINIMUM = 1e-10

def make_mating_pool(population_smiles: List[Mol], population_scores, pareto_table, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    all_tuples = list(zip(population_scores, population_smiles))
    
    # uniform distribution
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    print(all_tuples)
    print(population_probs)
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    
    # exponential prob
    # population_scores = [math.pow(10, s + MINIMUM) for s in population_scores]
    # sum_scores = sum(population_scores)
    # population_probs = [p / sum_scores for p in population_scores]
    # print(population_scores)
    # print(population_probs)
    # mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    
    
    # rank selection
    # weights = [1/(1+pareto_table[x]) for x in population_smiles]
    # summation = sum(weights)
    # weights = [w/summation for w in weights]
    # mating_indices = np.random.choice(len(all_tuples), size=offspring_size, p=weights, replace=True)
    
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    return mating_tuples

def reproduce(mating_tuples, mutation_rate, mol_lm=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent = []
    parent.append(random.choice(mating_tuples))
    parent.append(random.choice(mating_tuples))

    parent_mol = [t[1] for t in parent]
    new_child = co.crossover(parent_mol[0], parent_mol[1])
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child, new_child_mutation

def get_best_mol(population_scores, population_mol):
    top_mol = population_mol[np.argmax(population_scores)]
    top_smi = Chem.MolToSmiles(top_mol, canonical=True)
    return top_smi

class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None, config=None):
        super().__init__(args)
        self.model_name = "molleo"

        self.mol_lm = None
        if args.mol_lm == "GPT-4":
            self.mol_lm = GPT4()
        elif args.mol_lm == "GPT-oss":
            self.mol_lm = GPToss()
        elif args.mol_lm == "BioT5":
            self.mol_lm = BioT5()
        
        self.args = args
        self.config = config
        lm_name = "baseline"
        if args.mol_lm != None:
            lm_name = args.mol_lm
            self.mol_lm.task = self.args.task_mode

    def _optimize(self, config):

        self.oracle.assign_evaluator(self.args)

        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
    
        # Exploration run
        print(f"{str(len(self.all_smiles))} total SMILES")
        if self.args.starting == "zinc":
            population_smiles = list(np.random.choice(self.all_smiles, config["population_size"], replace=True))
        else:
            population_smiles = list(np.random.choice(self.all_smiles, config["population_size"], replace=False))

        print(f"{str(len(population_smiles))} SMILES selected")

        # select initial population
        print("Before sanitation: " + str(len(population_smiles)))
        population_smiles = self.sanitize(population_smiles)
        print("After sanitation: " + str(len(population_smiles)))
        self.oracle.starting_population = population_smiles.copy()
   
        population_scores = self.oracle(population_smiles)
        print(population_scores)

        patience = 0
        pareto_table = {}
        for smiles in population_smiles:
            pareto_table[smiles] = 0

        while True:
            if len(self.oracle) > 1:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
            else:
                old_score = 0
            # new_population
            mating_tuples = make_mating_pool(population_smiles, population_scores, pareto_table, config["population_size"])
            
            fp_scores = []
            offspring_mol_temp = []
            if self.args.mol_lm == 'GPT-4' or self.args.mol_lm == "GPT-oss":
                before = time.time()
                inputs = [
                    (idx, mating_tuples, config["mutation_rate"], self.oracle.min_evaluator,
                    self.oracle.max_evaluator, self.oracle.affin_cache,
                    self.args.single_parent, self.args.use_tools)
                    for idx in range(config["offspring_size"])
                ]

                gen_futures = {}   # future -> idx
                eval_futures = {}  # future -> (idx, smiles)
                print(set(self.oracle.min_evaluator))
                print(set(POSSIBLE_TARGETS))
                target = list(set([eva[0] for eva in self.oracle.min_evaluator]) & set(POSSIBLE_TARGETS))[0]
                print(target)
                offspring_smiles = []
                with ThreadPoolExecutor(max_workers=min(config["offspring_size"], 20)) as gen_pool, \
                    ThreadPoolExecutor(max_workers=min(config["offspring_size"], 20)) as eval_pool:

                    # Submit all generation tasks upfront
                    for inp in inputs:
                        idx = inp[0]
                        f = gen_pool.submit(self.mol_lm.edit, *inp)
                        gen_futures[f] = idx

                    # As each generation completes, immediately dispatch to evaluation.
                    # The gen_pool keeps running its remaining workers concurrently.
                    for gen_future in as_completed(gen_futures):
                        idx = gen_futures[gen_future]
                        try:
                            smiles = gen_future.result()
                        except Exception as e:
                            print(f"[Gen {idx}] Generation failed: {e}", flush=True)
                            smiles = None

                        if smiles is not None:
                            offspring_smiles.append(smiles)
                            eval_f = eval_pool.submit(calculate_boltz_nautilus, target, smiles, idx)
                            eval_futures[eval_f] = (idx, smiles)

                    # Collect evaluation results as they complete
                    results = {}  # idx -> (smiles, score)
                    for eval_future in as_completed(eval_futures):
                        idx, smiles = eval_futures[eval_future]
                        try:
                            score = eval_future.result()
                        except Exception as e:
                            print(f"[Eval {idx}] Evaluation failed: {e}", flush=True)
                            score = 0
                        self.oracle.affin_cache[target][smiles] = score
                after = time.time()
                print(f"LLM generation took {str(after-before)} seconds")
                    
                # offspring_smiles = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.oracle.min_evaluator, self.oracle.max_evaluator, self.oracle.affin_cache, single_parent=self.args.single_parent) for _ in range(config["offspring_size"])]
            elif self.args.mol_lm == 'BioT5':
                top_smi = get_best_mol(population_scores, population_smiles) 

                offspring_mol = [reproduce(mating_tuples, config["mutation_rate"]) for _ in range(config["offspring_size"])]
                offspring_mol = [item[0] for item in offspring_mol]
                editted_smi = []
                for m in offspring_mol:
                    if m != None:
                        editted_smi.append(Chem.MolToSmiles(m, canonical=True))
                ii = 0
                idxs = np.argsort(population_scores)[::-1]
                while len(editted_smi) < self.args.bin_size:
                    if ii == len(idxs):
                        print("exiting while loop before filling up bin..........")
                        break
                    m = population_smiles[idxs[ii]]
                    editted_mol = self.mol_lm.edit([m])[0]

                    if editted_mol != None:
                        s = Chem.MolToSmiles(editted_mol, canonical=True)
                        if s != None:
                            print("adding editted molecule!!!")
                            editted_smi.append(s)
                    ii += 1
                sim = get_fp_scores(editted_smi, top_smi)
                print("fp_scores_to_top", sim)
                sorted_idx = np.argsort(np.squeeze(sim))[::-1][:config["offspring_size"]]
                print("top 70", sorted_idx)
                editted_smi = np.array(editted_smi)[sorted_idx].tolist()
                offspring_mol = [Chem.MolFromSmiles(s) for s in editted_smi]
                print("len offspring_mol", len(offspring_mol))

            # add new_population
            print("Offspring size: " + str(len(offspring_smiles)))
            population_smiles += offspring_smiles
            population_smiles = self.sanitize(population_smiles)
            #Pareto optimal set
            # self.oracle.clean_buffer()
            print("Population size: " + str(len(population_smiles)))
            
            if not self.args.weighted_obj:
                before = time.time()
                population_smiles = list(self.oracle.select_pareto_front(population_smiles))
                after = time.time()
                print(f"Boltz calculation took {str(after-before)} seconds")
                pareto_table = {}
                for idx, smiles in enumerate(population_smiles):
                    pareto_table[smiles] = idx
                population_scores = self.oracle(population_smiles)
                population_tuples = list(zip(population_scores, population_smiles))
                population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
            else:
                population_scores = self.oracle(population_smiles)
                population_tuples = list(zip(population_scores, population_smiles))
                population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_smiles = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            print("Population Molecules: " + str(population_smiles))
            print("Population Scores: " + str(population_scores))

            ### early stopping
            # if len(self.oracle) > 1:
            #     self.sort_buffer()
            #     new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
            #     # import ipdb; ipdb.set_trace()
            #     if (new_score - old_score) < 1e-3:
            #         patience += 1
            #         if patience >= self.args.patience:
            #             self.log_intermediate(finish=True)
            #             print('convergence criteria met')
            #             # break
            #     else:
            #         patience = 0

            #     old_score = new_score
            
            print("Length of buffer: " + str(len(self.oracle.mol_buffer)))
            print("Max oracle calls: " + str(self.oracle.max_oracle_calls))
            if len(self.oracle.mol_buffer) >= self.oracle.max_oracle_calls:
                print("Finished")
                break

