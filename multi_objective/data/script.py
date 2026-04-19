from rdkit import Chem

ligs = []
with open("/home/ubuntu/MOLLEO/multi_objective/data/c-met.txt", 'r') as file:
    for line in file:
        ligs.append(line.strip())
        
new_smiles_list = []
smiles_set = set()
for smiles in ligs:
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
print(len(new_smiles_list))
