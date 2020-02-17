from __future__ import division, print_function
import argparse
import os
import numpy as np
import dynamic_time_warping as d

dat = np.loadtxt('ECG200_TRAIN.txt', delimiter =',')

normal = dat[dat[:, 0] == 1, 1:len(dat)]
abnormal = dat[dat[:, 0] == -1, 1:len(dat)]
num_comparisons = len(normal) * len(abnormal)

DIST_METRICS = {"manhattan": lambda t1, t2 :d.manhattan_dist(t1, t2),
                "dtw_0": lambda t1, t2: d.constrained_dtw(t1, t2, 0),
                "dtw_10": lambda t1, t2: d.constrained_dtw(t1, t2, 10),
                "dtw_25": lambda t1, t2: d.constrained_dtw(t1, t2, 25),
                "dtw_inf": lambda t1, t2: d.constrained_dtw(t1, t2, np.inf)}

def obtain_pairwise_distances(normal, abnormal, num_comparisons):

    """
    It receives two lists of vectors and computes all distance metrics between
    the vectors in the two lists.

    Returns a vector with average distances per metric. The positions in the
    vector match the metric names in DIST_METRICS.
    """
    avg_dist = np.zeros(len(DIST_METRICS), dtype=float)
    for i, metric in enumerate(sorted(DIST_METRICS)):

        # Iterate through all combinations of pairs of vectors
        for idx_g1 in range(len(normal)):
            for idx_g2 in range(len(abnormal)):
                t1 = normal[idx_g1]
                t2 = abnormal[idx_g2]

                # Determine which metric to compute by indexing dictionary
                dist = DIST_METRICS[metric](t1, t2)
                avg_dist[i] = avg_dist[i] + dist

    # Compute the average
    avg_dist = avg_dist / num_comparisons

    return avg_dist

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
parser.add_argument("--datadir", required=True,
                    help="Path to input directory containing the vectorized data")

parser.add_argument("--outdir", required=True,
                    help="Path to the output directory")
args = parser.parse_args()

# Create the output file and fill it with content as required by the specifications
file_name = "{}/timeseries_output.txt".format(args.outdir)

# If the output directory does not exist, then create it
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

with open(file_name, 'w') as f_out:

    # Compute all (average) distances between documents in the two groups
    dist = obtain_pairwise_distances(normal, abnormal, num_comparisons)

    # Transform the vector of distances to a string
    str_dist = str(dist)

    # Save the output
    f_out.write("{}:{}\t{}\n".format(normal, abnormal, str_dist))
