import scipy.io
import numpy as np
import argparse
import shortest_path_kernel as spk


mat = scipy.io.loadmat('/Users/damiano/Desktop/Master/Data_mining/Assignment_2/MUTAG.mat')
label = np.reshape(mat['lmutag'], (len(mat['lmutag'], )))
data = np.reshape(mat['MUTAG']['am'], (len(label), ))

idx_mut = label>0
mut = data[0:len(label[idx_mut])]

nonmut = data[len(label[idx_mut]):len(data)]

S1 = spk.floyd_warshall(mut)
S2 = spk.floyd_warshall(nonmut)

dist = spk.SPKernel(S1, S2)


# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
parser.add_argument("--datadir", required=True,
                    help="Path to input directory containing the vectorized data")

parser.add_argument("--outdir", required=True,
                    help="Path to the output directory")
args = parser.parse_args()

# Create the output file and fill it with content as required by the specifications
file_name = "{}/graphs_output.txt".format(args.outdir)
with open(file_name, 'w') as f_out:

    # Transform the vector of distances to a string
    str_dist = str(dist)

    # Save the output
    f_out.write("{}:{}\t{}\n".format(mut, nonmut, dist))
