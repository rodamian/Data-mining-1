import numpy as np
import argparse
import math


def strip_first_col(fname, delimiter=None):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
               yield line.split(delimiter, 1)[1]
            except IndexError:
               continue


def euclidean(x1, x2):
    return float(math.sqrt(np.sum((x1 - x2)**2)))


# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(
    description="compute k nearest neighbours distances")
parser.add_argument("--traindir", required=True,
                    help="Path to input directory containing the training data matrix_mirna_input.txt and phenotype.txt")
parser.add_argument("--testdir", required=True,
                    help="Path to the directory containing the test file matrix_mirna_input.txt and phenotype.txt ")
parser.add_argument("--mink", required=True, type = int,
                    help= 'minimum value of k')
parser.add_argument("--maxk", required=True, type = int,
                    help= 'maximum value of k')
parser.add_argument("--outdir", required=True,
                    help= 'path to the directory where the output file is stored')
args = parser.parse_args()

# Load data
train_data = np.loadtxt(strip_first_col("{}/matrix_mirna_input.txt".format(args.traindir)), skiprows=1)
test_data = np.loadtxt(strip_first_col("{}/matrix_mirna_input.txt".format(args.testdir)), skiprows=1)

args.maxk = args.maxk + 1

# Create distance matrix comparing every element in the test data and train data
dist = np.zeros(shape=(len(test_data), len(train_data)))

for i in range(len(test_data)):
    for j in range(len(train_data)):
        for k in range(len(test_data[0])):
            dist[i, j] = dist[i, j] + euclidean(train_data[j, k], test_data[i, k])

# Create matrix of first k neighbours indexes (NI)
idx = np.argsort(dist)
firstk_idx = idx[:, :args.maxk]

# Create list of identifiers
train_data_class = np.loadtxt("{}/phenotype.txt".format(args.traindir), skiprows=1, dtype=str)

# Predict class based on NI
positive = np.zeros(shape=(len(firstk_idx), args.maxk))
for i in range(len(positive)):
    for j in range(len(positive[0])):
        if train_data_class[firstk_idx[i, j], 1] == '+':
            positive[i, j] = positive[i, j-1] + 1
        else:
            positive[i, j] = positive[i, j-1]

# Create matrix of prediction results for test data (1 for '+' prediction, 0 for '-')
test_data_prediction = np.zeros(shape=(len(firstk_idx), args.maxk))

for i in range(len(positive)):
    for j in range(len(positive[0])):
        if positive[i, j] > len(positive[0:j])/2:
            test_data_prediction[i, j] = 1

        if positive[i, j] == len(positive[0:j])/2:
            test_data_prediction[i, j] = test_data_prediction[i, j-1]

        if positive[i, j] < len(positive[0:j])/2:
            test_data_prediction[i, j] = 0

# Compare between predicted labels and actual labels
test_data_class = np.loadtxt(strip_first_col('{}/phenotype.txt'.format(args.testdir)), skiprows=1, dtype=str)

# Create contingency table for every value of k
TP = np.zeros(len(test_data_prediction[0]))
TN = np.zeros(len(TP))
FP = np.zeros(len(TP))
FN = np.zeros(len(TP))
for k in range(len(TP)):
    for i in range(len(test_data_prediction)):
        if test_data_class[i] == '+' and test_data_prediction[i, k] == 1:
            TP[k] += 1
        if test_data_class[i] == '-' and test_data_prediction[i, k] == 1:
            FP[k] += 1
        if test_data_class[i] == '-' and test_data_prediction[i, k] == 0:
            TN[k] += 1
        if test_data_class[i] == '+' and test_data_prediction[i, k] == 0:
            FN[k] += 1

# Compute accuracy, precision and recall
acc = np.zeros(len(FP))
prec = np.zeros(len(FP))
rec = np.zeros(len(FP))

for k in range(len(FP)):

    acc[k] = (TN[k] + TP[k])/len(test_data_class)
    prec[k] = TP[k]/(TP[k] + FP[k])
    rec[k] = TP[k]/(TP[k] + FN[k])

# Create the output file
file_name = "{}/output_knn.txt".format(args.outdir)
with open(file_name, 'w') as f_out:
    f_out.write("Value of k\tAccuracy\tPrecision\tRecall\n")
    for k in range(args.mink, args.maxk):
        f_out.write("\n{}\t{:.2f}\t{:.2f}\t{:.2f}".format(k, acc[k], prec[k], rec[k]))
f_out.close()
