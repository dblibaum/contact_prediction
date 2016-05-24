from Bio.PDB import *
import re
import numpy as np
import cPickle as cp

true_inputs = []  # Container for true inputs

seq_lengths = []  # Container for sequence length of each input (for easy processing later)

# Create dictionary for aa 3gram embeddings
f = open("protVec_100d_3grams.csv")

aa_emb_dict = {}

for protvec in f:
    protvec = protvec[1:-2]
    values = protvec.split()
    vector = [float(x) for x in values[1:]]
    aa_emb_dict[values[0]] = vector

f.close()

#################################
# Process data for true samples #
#################################

f = open("pdbids.txt", "r")

rx = re.compile('\W+')

ctr = 0

for pdbid in f:

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

    seq_lengths.append(len(seq_embedding))

    # Make the full input
    full_input = seq_embedding + [[0 for x in range(100)]] + packed_indexes

    print full_input
    print len(full_input)
    print len(full_input[0])
    print len(full_input[-1])

    true_inputs.append(full_input)

f.close()

true_inp_plus_indexes = [true_inputs, seq_lengths]

f = open("true_cp_data.pkl")
cp.dump(true_inp_plus_indexes, f)
f.close()

# Shuffle the data
shuffled_indexes = np.random.permutation(len(true_inputs))
true_inputs = [true_inputs[i] for i in shuffled_indexes]

# Create train and test splits
true_inputs_train = true_inputs[5000:]
true_inputs_test = true_inputs[:5000]

###########################
# Create negative samples #
###########################

false_inputs_train = []
false_inputs_test = []

# Create negatives for train
inp_index = 0
for datum in true_inputs_train:
    sequence = datum[:seq_lengths[inp_index]]
    contacts = datum[seq_lengths[inp_index]+1:]

    shuffled_indexes = np.random.permutation(len(sequence))
    sequence = [sequence[i] for i in shuffled_indexes]
    shuffled_indexes = np.random.permutation(len(contacts))
    contacts = [contacts[i] for i in shuffled_indexes]

    full_input = sequence + [[0 for x in range(100)]] + contacts

    false_inputs_train.append(full_input)

    inp_index += 1

# Create negatives for test
inp_index = 0
for datum in true_inputs_test:
    sequence = datum[:seq_lengths[inp_index]]
    contacts = datum[seq_lengths[inp_index]+1:]

    shuffled_indexes = np.random.permutation(len(sequence))
    sequence = [sequence[i] for i in shuffled_indexes]
    shuffled_indexes = np.random.permutation(len(contacts))
    contacts = [contacts[i] for i in shuffled_indexes]

    full_input = sequence + [[0 for x in range(100)]] + contacts

    false_inputs_test.append(full_input)

    inp_index += 1

########################
# Create full datasets #
########################

# Create and shuffle full train dataset
all_inputs_train_samples = true_inputs_train + false_inputs_train

targets = [1 for _ in range(len(true_inputs_train))] + [0 for _ in range(len(false_inputs_train))]

shuffled_indexes = np.random.permutation(len(all_inputs_train_samples))

all_inputs_train_samples = [all_inputs_train_samples[i] for i in shuffled_indexes]
all_inputs_train_targets = [targets[i] for i in shuffled_indexes]

all_inputs_train = [all_inputs_train_samples, all_inputs_train_targets]

# Create and shuffle full test dataset
all_inputs_test_samples = true_inputs_test + false_inputs_test

targets = [1 for _ in range(len(true_inputs_test))] + [0 for _ in range(len(false_inputs_test))]

shuffled_indexes = np.random.permutation(len(all_inputs_test_samples))

all_inputs_test_samples = [all_inputs_test_samples[i] for i in shuffled_indexes]
all_inputs_test_targets = [targets[i] for i in shuffled_indexes]

all_inputs_test = [all_inputs_test_samples, all_inputs_test_targets]

# Pickle datasets
f = open("cp_train.pkl")
cp.dump(all_inputs_train, f)
f.close()

f = open("cp_test.pkl")
cp.dump(all_inputs_test, f)
f.close()