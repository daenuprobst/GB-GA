import sys
import time
import random
import sascorer
import numpy as np

from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')

class Gene:
  def __init__(self, size_mean, size_std, seed=-1):
    self.size_mean = size_mean
    self.size_std = size_std
    self.max_score = []
    self.count = 0

    if seed >= 0:
      random.seed(6)
      np.random.seed(6)

    self.initial_population = []

    self.logP_values = np.loadtxt('logP_values.txt')
    self.SA_scores = np.loadtxt('SA_scores.txt')
    self.cycle_scores = np.loadtxt('cycle_scores.txt')
    self.SA_mean =  np.mean(self.SA_scores)
    self.SA_std=np.std(self.SA_scores)
    self.logP_mean = np.mean(self.logP_values)
    self.logP_std= np.std(self.logP_values)
    self.cycle_mean = np.mean(self.cycle_scores)
    self.cycle_std=np.std(self.cycle_scores)

    # Precompute reactions
    self.rxn_crossover_non_ring = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
    self.rxn1_crossover_ring = [AllChem.ReactionFromSmarts('[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]'), AllChem.ReactionFromSmarts('[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]')]
    self.rxn2_crossover_ring = [AllChem.ReactionFromSmarts('([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]'), AllChem.ReactionFromSmarts('([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]')]
    

  def run(self, file_name, population_size, n_generations, mutation_rate):
    results = []
    size = []
    t0 = time.time()
    for i in range(10):
      self.max_score = [-99999.,'']
      self.count = 0

      population = self.seed_population(file_name, population_size)
      self.initial_population = population.copy()

      for _ in range(n_generations):
        fitness = self.calculate_normalized_fitness(population)
        mating_pool = self.make_mating_pool(population_size, population, fitness)
        population = self.reproduce(mating_pool, population_size, mutation_rate)

      print(i, self.max_score[0], self.max_score[1], AllChem.MolFromSmiles(self.max_score[1]).GetNumAtoms())
      results.append(self.max_score[0])
      size.append(AllChem.MolFromSmiles(self.max_score[1]).GetNumAtoms())

    t1 = time.time()
    print('')
    print('time ',t1-t0)
    print(max(results),np.array(results).mean(),np.array(results).std())
    print(max(size),np.array(size).mean(),np.array(size).std())

  def seed_population(self, file_name, population_size):
    mol_list = []

    with open(file_name, 'r') as file:
        for smiles in file:
          mol_list.append(AllChem.MolFromSmiles(smiles))

    return random.choices(mol_list, k=population_size)

  
  def make_mating_pool(self, population_size, population, fitness):
    mating_pool = []
    for _ in range(population_size):
      mating_pool.append(np.random.choice(population, p=fitness))

    return mating_pool
  

  def reproduce(self, mating_pool, population_size, mutation_rate):
    new_population = []
    for _ in range(population_size):
      parent_A = random.choice(mating_pool)
      parent_B = random.choice(mating_pool)

      new_child = self.crossover(parent_A,parent_B)

      if new_child != None:
        new_child = self.mutate(new_child,mutation_rate)

        if new_child != None:
          new_population.append(new_child)

    
    return new_population

  def calculate_normalized_fitness(self, population):
    fitness = []
    for i, g in enumerate(population):
      score = self.logP_score(g)

      # On error, draw a new molecule from the initial population
      if score == None:
        population[i] = random.choice(self.initial_population)

      fitness.append(max(float(score), 0.0))
    
    #calculate probability
    sum_fitness = sum(fitness)
    return [score/sum_fitness for score in fitness]

  def logP_score(self, mol):
    try:
      logp = Descriptors.MolLogP(mol)
    except:
      return None
      # print (mol, AllChem.MolToSmiles(mol))
      # sys.exit('failed to make a molecule')

    SA_score = -sascorer.calculateScore(mol)
    cycle_list = mol.GetRingInfo().AtomRings()

    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max(map(len, cycle_list))

    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6

    cycle_score = -cycle_length
    SA_score_norm = (SA_score - self.SA_mean) / self.SA_std
    logp_norm = (logp - self.logP_mean) / self.logP_std
    cycle_score_norm = (cycle_score - self.cycle_mean) / self.cycle_std
    score_one = SA_score_norm + logp_norm + cycle_score_norm
    
    self.count += 1
    if score_one > self.max_score[0]:
      self.max_score = [score_one, AllChem.MolToSmiles(mol)]
    
    
    return score_one

  def cut(self, mol):
    substructure = AllChem.MolFromSmarts('[*]-;!@[*]')
    
    if not mol.HasSubstructMatch(substructure): 
      return None
    
    bis = random.choice(mol.GetSubstructMatches(substructure)) #single bond not in ring
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = AllChem.FragmentOnBonds(mol,bs, addDummies=True, dummyLabels=[(1, 1)])
    fragments = AllChem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=False)

    for fragment in fragments:
      if AllChem.SanitizeMol(fragment, catchErrors=True) != 0:
        return None

    return fragments

  def cut_ring(self, mol):
    for _ in range(10):
      if random.random() < 0.5:
        substructure = AllChem.MolFromSmarts('[R]@[R]@[R]@[R]')
        if not mol.HasSubstructMatch(substructure): 
          return None
        bis = random.choice(mol.GetSubstructMatches(substructure))
        bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
      else:
        substructure = AllChem.MolFromSmarts('[R]@[R;!D2]@[R]')
        if not mol.HasSubstructMatch(substructure): 
          return None
        bis = random.choice(mol.GetSubstructMatches(substructure))
        bis = ((bis[0],bis[1]),(bis[1],bis[2]),)
      
      #print bis
      bs = [mol.GetBondBetweenAtoms(x,y).GetIdx() for x,y in bis]

      fragments_mol = AllChem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1),(1,1)])

      # Is result the same when sanitizeFrags = False?
      fragments = AllChem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=False)

      if len(fragments) == 2:
        for fragment in fragments:
          if AllChem.SanitizeMol(fragment, catchErrors=True) != 0:
            pass
        return fragments
      
    return None

  def ring_OK(self, mol):
    if not mol.HasSubstructMatch(AllChem.MolFromSmarts('[R]')):
      return True
    
    ring_allene = mol.HasSubstructMatch(AllChem.MolFromSmarts('[R]=[R]=[R]'))
    
    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max(map(len, cycle_list))
    macro_cycle = max_cycle_length > 6
    
    double_bond_in_small_ring = mol.HasSubstructMatch(AllChem.MolFromSmarts('[r3,r4]=[r3,r4]'))
    
    return not (ring_allene or macro_cycle or double_bond_in_small_ring)

  def mol_OK(self, mol):
    try:
      AllChem.SanitizeMol(mol)
      target_size = self.size_std * np.random.randn() + self.size_mean
      if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
        return True
      else:
        return False
    except:
      return False

  def crossover_ring(self, parent_A, parent_B):
    ring_smarts = AllChem.MolFromSmarts('[R]')
    if not (parent_A.HasSubstructMatch(ring_smarts) or parent_B.HasSubstructMatch(ring_smarts)):
      return None

    for _ in range(10):
      fragments_A = self.cut_ring(parent_A)
      fragments_B = self.cut_ring(parent_B)

      if fragments_A == None or fragments_B == None:
        return None
      
      new_mol_trial = []
      for rxn in self.rxn1_crossover_ring:
        new_mol_trial = []
        for fa in fragments_A:
          for fb in fragments_B:
            new_mol_trial.append(rxn.RunReactants((fa, fb))[0]) 

      new_mols = []
      for rxn in self.rxn2_crossover_ring:
        for m in new_mol_trial:
          m = m[0]
          if self.mol_OK(m):
            new_mols += list(rxn.RunReactants((m,)))
      
      new_mols2 = []
      for m in new_mols:
        m = m[0]
        if self.mol_OK(m) and self.ring_OK(m):
          new_mols2.append(m)
      
      if len(new_mols2) > 0:
        return random.choice(new_mols2)
      
    return None

  def crossover_non_ring(self, parent_A, parent_B):
    for _ in range(10):
      fragments_A = self.cut(parent_A)
      fragments_B = self.cut(parent_B)

      if fragments_A == None or fragments_B == None:
        return None

      new_mol_trial = []

      for fa in fragments_A:
        for fb in fragments_B:
          new_mol_trial.append(self.rxn_crossover_non_ring.RunReactants((fa, fb))[0]) 
                                  
      new_mols = []
      for mol in new_mol_trial:
        mol = mol[0]
        if self.mol_OK(mol):
          new_mols.append(mol)
      
      if len(new_mols) > 0:
        return random.choice(new_mols)
      
    return None

  def crossover(self, parent_A, parent_B):
    parent_smiles = [AllChem.MolToSmiles(parent_A), AllChem.MolToSmiles(parent_B)]
    try:
      AllChem.Kekulize(parent_A, clearAromaticFlags=True)
      AllChem.Kekulize(parent_B, clearAromaticFlags=True)
    except:
      pass

    for _ in range(10):
      if random.random() <= 0.5:
        new_mol = self.crossover_non_ring(parent_A, parent_B)
        if new_mol != None:
          new_smiles = AllChem.MolToSmiles(new_mol)
        if new_mol != None and new_smiles not in parent_smiles:
          return new_mol
      else:
        new_mol = self.crossover_ring(parent_A,parent_B)
        if new_mol != None:
          new_smiles = AllChem.MolToSmiles(new_mol)
        if new_mol != None and new_smiles not in parent_smiles:
          return new_mol
    
    return None

  def delete_atom(self, mol=None):
    choices = ['[*:1]~[D1:2]>>[*:1]', '[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]',
              '[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]',
              '[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]',
              '[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]']
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]
    
    return np.random.choice(choices, p=p)

  def append_atom(self, mol=None):
    choices = [['single', ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'], 7 * [1.0 / 7.0]],
              ['double', ['C', 'N', 'O'], 3 * [1.0 / 3.0]],
              ['triple', ['C', 'N'],2 * [1.0 / 2.0]] ]
    p_BO = [0.6, 0.35, 0.05]
    
    index = np.random.choice(list(range(3)), p=p_BO)
    
    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)
    
    if BO == 'single': 
      rxn_smarts = '[*;!H0:1]>>[*:1]-' + new_atom
    if BO == 'double': 
      rxn_smarts = '[*;!H0;!H1:1]>>[*:1]=' + new_atom
    if BO == 'triple': 
      rxn_smarts = '[*;H3:1]>>[*:1]#' + new_atom
      
    return rxn_smarts

  def insert_atom(self, mol=None):
    choices = [['single', ['C', 'N', 'O', 'S'], 4 * [1.0 / 4.0]],
              ['double', ['C', 'N'], 2 * [1.0 / 2.0]],
              ['triple', ['C'], [1.0]]]

    p_bonds = [0.6, 0.35, 0.05]
    
    index = np.random.choice(range(3), p=p_bonds)
    
    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)
    
    if BO == 'single': 
      rxn_smarts = '[*:1]~[*:2]>>[*:1]' + new_atom + '[*:2]'
    if BO == 'double': 
      rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=' + new_atom + '-[*:2]'
    if BO == 'triple': 
      rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#' + new_atom + '-[*:2]'
      
    return rxn_smarts

  def change_bond_order(self, mol=None):
    choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]','[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
              '[*:1]#[*:2]>>[*:1]=[*:2]','[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
    p = [0.45, 0.45, 0.05, 0.05]
    
    return np.random.choice(choices, p=p)

  def delete_cyclic_bond(self, mol=None):
    return '[*:1]@[*:2]>>([*:1].[*:2])'

  def add_ring(self, mol=None):
    choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
              '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
              '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
              '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1'] 
    p = [0.05, 0.05, 0.45, 0.45]
    
    return np.random.choice(choices, p=p)

  def change_atom(self, mol):
    choices = ['#6','#7','#8','#9','#16','#17','#35']
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
    
    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(AllChem.MolFromSmarts('[' + X + ']')):
      X = np.random.choice(choices, p=p)

    Y = np.random.choice(choices, p=p)
    while Y == X:
      Y = np.random.choice(choices, p=p)
    
    return '[' + X + ':1]>>[' + Y + ':1]'

  def mutate(self, mol, mutation_rate):

    if random.random() > mutation_rate:
      return mol
    
    AllChem.Kekulize(mol, clearAromaticFlags=True)
    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]

    for _ in range(10):
      reaction_methods = {
        0: self.insert_atom,
        1: self.change_bond_order,
        2: self.delete_cyclic_bond,
        3: self.add_ring,
        4: self.delete_atom,
        5: self.change_atom,
        6: self.append_atom
      }

      rxn_smarts = reaction_methods[np.random.choice(range(7), p=p)](mol) 
      
      rxn = AllChem.ReactionFromSmarts(rxn_smarts)

      new_mol_trial = rxn.RunReactants((mol,))
      
      new_mols = []
      for m in new_mol_trial:
        m = m[0]
        if self.mol_OK(m) and self.ring_OK(m):
          new_mols.append(m)
      
      if len(new_mols) > 0:
        return random.choice(new_mols)
    
    return None