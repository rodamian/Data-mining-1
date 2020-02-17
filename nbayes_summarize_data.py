import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--traindir", required=True,
                    help="Path to input directory containing the training data tumor_info.txt")
parser.add_argument("--outdir", required=True,
                    help="Path to the directory where the output text will be saved")
args = parser.parse_args()

# Loading the data
train_data = pd.read_csv("{}/tumor_info.txt".format(args.traindir), sep='\t', header=None, dtype=float)
train_data2 = train_data.ix[train_data.values[:, 4] == 2]
train_data4 = train_data.ix[train_data.values[:, 4] == 4]

# Compute value frequency for every feature in both classes
freq2 = train_data2.apply(pd.value_counts).fillna(0)/(len(train_data2) - train_data2.isnull().sum())
freq4 = train_data4.apply(pd.value_counts).fillna(0)/(len(train_data4) - train_data4.isnull().sum())
freq2 = freq2.drop(columns=4)
freq4 = freq4.drop(columns=4)
freq2.reset_index(level=0, inplace=True)
freq4.reset_index(level=0, inplace=True)
freq2.columns = ("Values", "clump", "uniformity", "marginal", "mitoses")
freq4.columns = ("Values", "clump", "uniformity", "marginal", "mitoses")
freq2 = freq2.round(3)
freq4 = freq4.round(3)

# Create the output files
freq2.to_csv(r"{}/output_summary_class_2.txt".format(args.outdir), sep='\t', index=False)
freq4.to_csv(r"{}/output_summary_class_4.txt".format(args.outdir), sep='\t', index=False)
