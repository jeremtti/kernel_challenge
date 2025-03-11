import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
import csv
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from itertools import product
from scipy.sparse import coo_matrix, csr_matrix
from joblib import Parallel, delayed
from scipy.sparse.linalg import spsolve


NUCL = set('ACGT')

def generate_mismatch_neighbors(kmer, m, alphabet=('A', 'C', 'G', 'T')):
    """
    Generate all k-mers that differ from `kmer` in at most `m` positions.
    """
    
    results = []
    k_len = len(kmer)

    def backtrack(prefix, idx, mismatches_used):
        if idx == k_len:
            results.append(prefix)
            return
        current_char = kmer[idx]
        for letter in alphabet:
            cost = (letter != current_char)
            if mismatches_used + cost <= m:
                backtrack(prefix + letter, idx + 1, mismatches_used + cost)

    backtrack("", 0, 0)
    return results

def process_sequence(args):
    i, seq, k, m, pattern_to_index = args
    row = []
    col = []
    data = []
    seq_len = len(seq)
    for start in range(seq_len - k + 1):
        original_kmer = seq[start:start+k]
        neighbors = generate_mismatch_neighbors(original_kmer, m, NUCL)
        for nb in neighbors:
            if nb in pattern_to_index:
                row.append(i)
                col.append(pattern_to_index[nb])
                data.append(1)
    return row, col, data

def mismatch_kernel_matrix_sparse(X, k, m, n_jobs=-1):
    """
    Build an N x (4^k) feature matrix for the mismatch kernel.
    """
    all_kmers = [''.join(p) for p in product(NUCL, repeat=k)]
    pattern_to_index = {km: i for i, km in enumerate(all_kmers)}
    dim = len(all_kmers)

    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sequence)((i, seq, k, m, pattern_to_index))
        for i, seq in enumerate(tqdm(X))
    )

    row = []
    col = []
    data = []

    for r in results:
        row.extend(r[0])
        col.extend(r[1])
        data.extend(r[2])

    mm_coo = coo_matrix((data, (row, col)), shape=(len(X), dim), dtype=np.float32)
    mm_coo.sum_duplicates()
    return mm_coo.tocsr()

def kernel_ridge_regression(K, Y, lambd, sparse=True):
    n = K.shape[0]
    if sparse:
        return spsolve(K + n * lambd * csr_matrix(np.eye(n)), Y)
    return np.linalg.solve(K + n * lambd * np.eye(n), Y)

def krr_fit_transform_new(Xsubtr_feat, Ysubtr, Xsubva_feat, lam):
    """
    Fits kernel ridge on precomputed feature matrices Xsubtr_feat, Xsubva_feat.
    
    """
    
    Xsubtr_kernel = Xsubtr_feat @ Xsubtr_feat.T
    alpha = kernel_ridge_regression(Xsubtr_kernel, Ysubtr, lam, sparse=True)
    K_val = Xsubtr_feat @ Xsubva_feat.T
    Ysubva_pred = (K_val.T @ alpha) >= 0.5
    Ysubtr_pred = (Xsubtr_kernel @ alpha) >= 0.5

    return Ysubtr_pred, Ysubva_pred

#load csv files
Xtr0 = pd.read_csv('data/Xtr0.csv')
Xte0 = pd.read_csv('data/Xte0.csv')
Ytr0 = pd.read_csv('data/Ytr0.csv')

Xtr1 = pd.read_csv('data/Xtr1.csv')
Xte1 = pd.read_csv('data/Xte1.csv')
Ytr1 = pd.read_csv('data/Ytr1.csv')

Xtr2 = pd.read_csv('data/Xtr2.csv')
Xte2 = pd.read_csv('data/Xte2.csv')
Ytr2 = pd.read_csv('data/Ytr2.csv')

#dictionnarys for each dataset
Xtr = {0:Xtr0, 1:Xtr1, 2:Xtr2}
Xte = {0:Xte0, 1:Xte1, 2:Xte2}
Ytr = {0:Ytr0, 1:Ytr1, 2:Ytr2}


Xtr0_list = Xtr0['seq'].tolist()
Xte0_list = Xte0['seq'].tolist()
Ytr0_list = Ytr0['Bound'].values

Xtr1_list = Xtr1['seq'].tolist()
Xte1_list = Xte1['seq'].tolist()
Ytr1_list = Ytr1['Bound'].values

Xtr2_list = Xtr2['seq'].tolist()
Xte2_list = Xte2['seq'].tolist()
Ytr2_list = Ytr2['Bound'].values

print("Kernels for dataset 0...")
k1, k2 = 9, 2
rho = 1
lam = 3
Xtr0_feat1 = mismatch_kernel_matrix_sparse(Xtr0_list, k=k1, m=1)
Xte0_feat1 = mismatch_kernel_matrix_sparse(Xte0_list, k=k1, m=1)
Xtr0_feat2 = mismatch_kernel_matrix_sparse(Xtr0_list, k=k2, m=1)
Xte0_feat2 = mismatch_kernel_matrix_sparse(Xte0_list, k=k2, m=1)
Xtr0_feat2 = Xtr0_feat2.astype(float)
Xte0_feat2 = Xte0_feat2.astype(float)
Xtr0_feat2.data *= rho
Xte0_feat2.data *= rho
Xtr0_feat = hstack((Xtr0_feat1, Xtr0_feat2)).tocsr()
Xte0_feat = hstack((Xte0_feat1, Xte0_feat2)).tocsr()
Ytr0_pred, Yte0_pred = krr_fit_transform_new(Xtr0_feat, Ytr0_list, Xte0_feat, lam=lam)

print("Kernels for dataset 1...")
k1, k2 = 10, 2
rho = 0.3
lam = 0.001
Xtr1_feat1 = mismatch_kernel_matrix_sparse(Xtr1_list, k=k1, m=1)
Xte1_feat1 = mismatch_kernel_matrix_sparse(Xte1_list, k=k1, m=1)
Xtr1_feat2 = mismatch_kernel_matrix_sparse(Xtr1_list, k=k2, m=1)
Xte1_feat2 = mismatch_kernel_matrix_sparse(Xte1_list, k=k2, m=1)
Xtr1_feat2 = Xtr1_feat2.astype(float)
Xte1_feat2 = Xte1_feat2.astype(float)
Xtr1_feat2.data *= rho
Xte1_feat2.data *= rho
Xtr1_feat = hstack((Xtr1_feat1, Xtr1_feat2)).tocsr()
Xte1_feat = hstack((Xte1_feat1, Xte1_feat2)).tocsr()
Ytr1_pred, Yte1_pred = krr_fit_transform_new(Xtr1_feat, Ytr1_list, Xte1_feat, lam=lam)

print("Kernels for dataset 2...")
k1, k2 = 10, 8
rho = 1
lam = 0.6
Xtr2_feat1 = mismatch_kernel_matrix_sparse(Xtr2_list, k=k1, m=1)
Xte2_feat1 = mismatch_kernel_matrix_sparse(Xte2_list, k=k1, m=1)
Xtr2_feat2 = mismatch_kernel_matrix_sparse(Xtr2_list, k=k2, m=1)
Xte2_feat2 = mismatch_kernel_matrix_sparse(Xte2_list, k=k2, m=1)
Xtr2_feat2 = Xtr2_feat2.astype(float)
Xte2_feat2 = Xte2_feat2.astype(float)
Xtr2_feat2.data *= rho
Xte2_feat2.data *= rho
Xtr2_feat = hstack((Xtr2_feat1, Xtr2_feat2)).tocsr()
Xte2_feat = hstack((Xte2_feat1, Xte2_feat2)).tocsr()
Ytr2_pred, Yte2_pred = krr_fit_transform_new(Xtr2_feat, Ytr2_list, Xte2_feat, lam=lam)

with open('Yte.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'Bound']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, y in enumerate(Yte0_pred):
        writer.writerow({'id': i, 'Bound': int(y)})
    for i, y in enumerate(Yte1_pred, start=i+1):
        writer.writerow({'id': i, 'Bound': int(y)})
    for i, y in enumerate(Yte2_pred, start=i+1):
        writer.writerow({'id': i, 'Bound': int(y)})

print("Done")