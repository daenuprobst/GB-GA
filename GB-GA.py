from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from gene import Gene

import numpy as np
import random
import time
import sys

population_size = 20 
generations = 50
mutation_rate = 0.01

gene = Gene(39.15, 3.5)


print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('average_size/size_stdev', gene.size_mean, gene.size_std)
print('')

file_name = 'ZINC_first_1000.smi'

gene.run(file_name, population_size, generations, mutation_rate)