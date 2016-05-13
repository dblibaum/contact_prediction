from Bio.PDB import *
import re

f = open("pdbids.txt", "r")

rx = re.compile('\W+')

ctr = 0

for pdbid in f:

    pdbid = rx.sub(' ', pdbid).strip()

    pdbl = PDBList()

    try:
        pdbl.retrieve_pdb_file(pdbid, pdir="pdb_files")
    except IOError:
        print "Could not download " + pdbid + "."

    ctr += 1

    print str(ctr)

# pdbl = PDBList()

# pdbl.retrieve_pdb_file("101M", pdir="pdb_files")