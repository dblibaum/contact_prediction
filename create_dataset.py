from Bio.PDB import *
import re
import numpy as np

all_inputs = []

# Create dictionary for aa 3gram embeddings
f = open("protVec_100d_3grams.csv")

aa_emb_dict = {}

for protvec in f:
    protvec = protvec[1:-2]
    values = protvec.split()
    vector = [float(x) for x in values[1:]]
    aa_emb_dict[values[0]] = vector

f.close()

f = open("pdbids.txt", "r")

rx = re.compile('\W+')

ctr = 0

# for pdbid in f:

pdbid = "1A0B"

pdbid = rx.sub(' ', pdbid).strip()

parser = PDBParser()

structure = parser.get_structure(pdbid, "pdb_files/pdb" + pdbid.lower() + ".ent")

res_list = Selection.unfold_entities(structure, 'R')

res_list = [i for i in res_list if i.has_id("CA")]

contact_matrix = np.zeros((len(res_list), len(res_list)))

# Build contact matrix
i = 0
for residue_i in res_list:
    ca_i = residue_i['CA']
    j = 0
    for residue_j in res_list:
        try:
            ca_j = residue_j['CA']
        except KeyError:
            print "No CA for residue " + str(residue_j)
            continue
        distance = abs(ca_i-ca_j)
        if distance < 8:
            contact_matrix[i][j] = 1
        else:
            contact_matrix[i][j] = 0
        j += 1
    i += 1

packed_contacts = np.zeros(len(res_list)**2 / 2)

# Create packed matrix representation (U)
for i in range(len(res_list)):
    for j in range(len(res_list)):
        if i <= j:
            packed_contacts[+j*(j-1)/2] = contact_matrix[i][j]

packed_indexes = []

# Create further packed representation by making an array of pointers to contacts
# The indexes are stored in a binary representation padded to 100 bits, to match sequence embedding
for i in range(len(packed_contacts)):
    if packed_contacts[i] == 1:
        bin_array = []
        bin = "{0:b}".format(i).zfill(100)
        for char in bin:
            bin_array.append(int(char))
        packed_indexes.append(bin_array)

# Get the sequence
ppb = CaPPBuilder()
seq = str(ppb.build_peptides(structure)[0].get_sequence())

# Create embedding
seq_embedding = []

for i in range(len(seq) - 2):
    three_gram = seq[i:i+3]
    vector = aa_emb_dict[three_gram]
    seq_embedding.append(vector)

# Make the full input
full_input = seq_embedding + [[0 for x in range(100)]] + packed_indexes

print full_input
print len(full_input)
print len(full_input[0])
print len(full_input[-1])

all_inputs.append(full_input)