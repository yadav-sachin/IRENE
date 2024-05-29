import torch
import numpy as np
from tqdm import trange
from xclib.evaluation import xc_metrics
import numba as nb
from xclib.utils.matrix import SMatrix
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import hnswlib
import functools
import time
import pandas as pd
import argparse
from collections import defaultdict
import torch.distributed as dist
from sklearn.preprocessing import normalize
from xclib.utils.sparse import csr_from_arrays
import xclib.evaluation.xc_metrics as xc_metrics
from xclib.utils.shortlist import Shortlist
from tqdm import tqdm


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


from contextlib import contextmanager
@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

@timeit
def get_lbl_embeddings(tokenization_folder, prefix, num_Z, model, max_len, label_mapping, bsz=2000):
    """Get embeddings for given tokenized files"""
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', shape=(num_Z, max_len), dtype=np.int64)
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_attention_mask.dat",
        mode='r', shape=(num_Z, max_len), dtype=np.int64)
    with evaluating(model), torch.no_grad():
        for i in range(0, num_Z, bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            lbl_ids = torch.LongTensor(label_mapping[i: i + len(batch_attention_mask)])
            _batch_embeddings = model(
                None, None, batch_input_ids, batch_attention_mask, lbl_ids).cpu().numpy()
            if(i == 0):
                embeddings = np.zeros((num_Z, _batch_embeddings.shape[1]))
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings
    return embeddings


@timeit
def get_lbl_representations(
    tokenization_folder,
    prefix,
    neighbors_lbl_indices,
    self_and_neighbors_attn_mask,
    num_Z,
    model,
    max_len,
    bsz=2000
):
    """Get Label Representations for given tokeized files"""
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode="r",
        shape=(num_Z, max_len),
        dtype=np.int64,
    )
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_attention_mask.dat",
        mode="r",
        shape=(num_Z, max_len),
        dtype=np.int64
    )

    with evaluating(model), torch.no_grad():
        for i in range(0, num_Z, bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            batch_neighbors_lbl_indices = torch.LongTensor(neighbors_lbl_indices[i: i + bsz])
            batch_self_and_neighbors_attn_mask = torch.from_numpy(self_and_neighbors_attn_mask[i: i + bsz]).bool()
            _batch_embeddings = (
                model(
                    doc_input_ids=None,
                    doc_attention_mask=None,
                    lbl_input_ids=batch_input_ids,
                    lbl_attention_mask=batch_attention_mask,
                    neighbors_index=batch_neighbors_lbl_indices,
                    self_and_neighbors_attention_mask=batch_self_and_neighbors_attn_mask,
                )
                .cpu()
                .numpy()
            )
            if i == 0:
                embeddings = np.zeros((num_Z, _batch_embeddings.shape[1]))
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings
    return embeddings


@timeit
def get_doc_embeddings(tokenization_folder, prefix, num_Z, model, max_len, bsz=2000):
    """Get embeddings for given tokenized files"""
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', shape=(num_Z, max_len), dtype=np.int64)
    attention_mask = np.memmap(
        f"{tokenization_folder}/{prefix}_attention_mask.dat",
        mode='r', shape=(num_Z, max_len), dtype=np.int64)
    with evaluating(model), torch.no_grad():
        for i in range(0, num_Z, bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz])
            batch_attention_mask = torch.LongTensor(attention_mask[i: i + bsz])
            _batch_embeddings = model(
                batch_input_ids, batch_attention_mask, None, None, None).cpu().numpy()
            if(i == 0):
                embeddings = np.zeros((num_Z, _batch_embeddings.shape[1]))
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings
    return embeddings

@timeit
def get_embeddings_w_clf(tokenization_folder, prefix, model, max_len, label_mapping, bsz=2000):
    """Get embeddings for given tokenized files"""
    lengths = np.load(f"{tokenization_folder}/{prefix}_lengths.npy")
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', shape=(len(lengths), max_len), dtype=np.int32)
    with evaluating(model), torch.no_grad():
        for i in range(0, len(lengths), bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz].astype(np.int64))
            batch_am = np.zeros(batch_input_ids.shape, dtype=np.int64)
            for j in range(batch_input_ids.shape[0]):
                batch_am[j][:lengths[i + j]] = 1
            batch_attention_mask = torch.LongTensor(batch_am)
            lbl_ids = torch.LongTensor(label_mapping[i: i + len(batch_attention_mask)])
            _batch_embeddings = model(
                None, None, batch_input_ids, batch_attention_mask, lbl_ids).cpu().numpy()
            if(i == 0):
                embeddings = np.zeros((len(lengths), _batch_embeddings.shape[1]))
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings
    return embeddings


@timeit
def get_bert_embeddings(tokenization_folder, prefix, model, max_len, bsz=2000):
    """Get embeddings for given tokenized files"""
    lengths = np.load(f"{tokenization_folder}/{prefix}_lengths.npy")
    input_ids = np.memmap(
        f"{tokenization_folder}/{prefix}_input_ids.dat",
        mode='r', shape=(len(lengths), max_len), dtype=np.int32)
    with evaluating(model), torch.no_grad():
        for i in range(0, len(lengths), bsz):
            batch_input_ids = torch.LongTensor(input_ids[i: i + bsz].astype(np.int64))
            batch_am = np.zeros(batch_input_ids.shape, dtype=np.int64)
            for j in range(batch_input_ids.shape[0]):
                batch_am[j][:lengths[i + j]] = 1
            batch_attention_mask = torch.LongTensor(batch_am)
            _batch_embeddings = model(
                batch_input_ids, batch_attention_mask, None, None, None).cpu().numpy()
            if(i == 0):
                embeddings = np.zeros((len(lengths), _batch_embeddings.shape[1]))
            embeddings[i: i + batch_input_ids.shape[0]] = _batch_embeddings
    return embeddings


def get_filter_map(fname):
    """Load filter file as numpy array"""
    if fname is not None and fname != "":
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred, mapping):
    """Filter predictions using given mapping"""
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def evaluate(_true, _pred, _train, k, A, B, recall_only=False):
    """Evaluate function
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    _true.indices = _true.indices.astype('int64')
    if not recall_only:
        inv_propen = xc_metrics.compute_inv_propesity(_train, A, B)
        acc = xc_metrics.Metrics(_true, inv_propen)
        acc = acc.eval(_pred, 5)
    else:
        print("Only R@k is computed. Don't be surprised with 0 val of others")
        acc = np.zeros((4, 5))
    p = xc_metrics.format(*acc)
    rec = xc_metrics.recall(_pred, _true, k)  # get the recall
    return acc, rec


def evaluate_with_filter(true_labels, predicted_labels,
                         train_labels, filter_labels, k,
                         A, B, recall_only):
    """Evaluate function with support of filter file
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    mapping = get_filter_map(filter_labels)
    predicted_labels = filter_predictions(predicted_labels, mapping)
    return evaluate(
        true_labels, predicted_labels, train_labels, k, A, B, recall_only)


def _predict_anns(X, clf, k, M=100, efC=300):
    """
    Train a nearest neighbor structure on label embeddings
    - for a given test point: query the graph for closest label
    - HNSW graph would return cosine distance between and document and
    """
    num_instances, num_labels = len(X), len(clf)
    graph = Shortlist(
        method='hnswlib', M=M, efC=efC, efS=k,
        num_neighbours=k, space='cosine', num_threads=64)    
    print("Training ANNS")
    graph.fit(clf)
    print("Predicting using ANNS")
    ind, sim = graph.query(X)
    pred = csr_from_arrays(ind, sim, (num_instances, num_labels))
    return pred


def _predict_ova(X, clf, k=20, batch_size=32, device="cuda", return_sparse=True):
    """Predictions in brute-force manner"""
    torch.set_grad_enabled(False)
    num_instances, num_labels = len(X), len(clf)
    batches = np.array_split(range(num_instances), num_instances//batch_size)
    output = SMatrix(
        n_rows=num_instances,
        n_cols=num_labels,
        nnz=k)
    X = torch.from_numpy(X)        
    clf = torch.from_numpy(clf).to(device).T   
    for ind in tqdm(batches):
        s_ind, e_ind = ind[0], ind[-1] + 1
        _X = X[s_ind: e_ind].to(device)
        ans = _X @ clf
        vals, ind = torch.topk(
            ans, k=k, dim=-1, sorted=True)
        output.update_block(
            s_ind, ind.cpu().numpy(), vals.cpu().numpy())
        del _X
    if return_sparse:
        return output.data()
    else:
        return output.data('dense')[0]


def predict_and_eval(features, clf, labels,
                     trn_labels, filter_labels,
                     A, B, k=10, mode='ova', huge=False):
    """
    Predict on validation set and evaluate
    * support for filter file (pass "" or empty file otherwise)
    * ova will get top-k predictions but anns would get 300 (change if required)"""
    mode='anns' if huge else mode
    if mode == 'ova':
        pred = _predict_ova(normalize(features, copy=True), normalize(clf, copy=True), k=k)
    else:
        pred = _predict_anns(features, clf, k=300)
    labels.indices = labels.indices.astype('int64')
    acc, r = evaluate_with_filter(labels, pred, trn_labels, filter_labels, k, A, B, huge)
    return acc, r, pred