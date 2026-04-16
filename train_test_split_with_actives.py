import random
import os

import random
from collections import defaultdict


protein_lines = defaultdict(list)

with open("all_with_actives.types") as f:
    for line in f:
        parts = line.strip().split()
        protein_id = parts[4]
        protein_lines[protein_id].append(line)


all_proteins = list(protein_lines.keys())
random.shuffle(all_proteins)

train_ratio = 0.8
split_index = int(len(all_proteins) * train_ratio)

train_proteins = set(all_proteins[:split_index])
test_proteins = set(all_proteins[split_index:])


with open("train_actives.types", "w") as train_f, open("test_actives.types", "w") as test_f:
    for prot in all_proteins:
        lines = protein_lines[prot]
        if prot in train_proteins:
            train_f.writelines(lines)
        else:
            test_f.writelines(lines)

