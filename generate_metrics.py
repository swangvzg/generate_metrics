import pandas as pd
import numpy as np
import squidpy as sq
import scanpy as sc

import os
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc_context
import scipy
import piecewise_regression
from skimage import filters
from sklearn.mixture import GaussianMixture
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csr_matrix
from scipy.stats import ttest_ind
import datetime
import time
import utils
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_detected_trs", required=True, help="path to detected_trs.csv")
    parser.add_argument("--path_cell_by_gene", required=True, help="path to cell_by_gene.csv")
    parser.add_argument("--path_cell_metadata", required=True, help="path to cell_metadata.csv")
    parser.add_argument("--path_code_book", required=True, help="path to codebook.csv")
    parser.add_argument("--exp_id", required=True, help="experiment ID")
    parser.add_argument("--output_dir", required=True, help="path to output directory")
    parser.add_argument("--exp_data", required=False, help="path to expression csv file")
    parser.add_argument("--fov_size", required=False, help="fov size, default is 0.202, 0.298 for M1.7", default=0.202)
    parser.add_argument("--portion", required=False, help="Random select portion of data for analysis, default is 0.1 (10%)", default=0.1)

    args = parser.parse_args()

    print(args.fov_size)

    if args.fov_size not in ['0.202', '0.298']:
        sys.exit("Error: fov size has to be either 0.202 or 0.298.")

    exp = pd.read_csv(args.exp_data,index_col=0)

    # check if output dir exists
    if not os.path.exists(args.output_dir) or not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    st = time.time()

    res1 = utils.calc_qc_metrics(
        args.path_detected_trs, 
        args.path_cell_by_gene, 
        args.path_cell_metadata,
        fov_size = float(args.fov_size),
        exp_table = exp,
        path_to_codebook = args.path_code_book,
        outputfolderpath=args.output_dir,
        expid=args.exp_id,
        portion=float(args.portion),
        plot=True)

    et=time.time()
    el = (et-st)/60
    print(f"{el} mins used.")

if __name__ == '__main__':
    main()