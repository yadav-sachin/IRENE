import torch
import os
import numpy as np
from tqdm import trange
from xclib.evaluation import xc_metrics
import numba as nb
from xclib.utils.matrix import SMatrix
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import hnswlib
import gc
import functools
import time
import pandas as pd
from collections import defaultdict
import torch.distributed as dist
from sklearn.preprocessing import normalize
from xclib.utils.sparse import csr_from_arrays
import xclib.evaluation.xc_metrics as xc_metrics
from xclib.utils.shortlist import Shortlist
from tqdm import tqdm

from xclib.data import data_utils

np.int = int
np.float = float
np.bool = bool

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


@timeit
def get_lbl_representations_zsxc_scores_wo_encoder(
    unnorm_lbl_embeddings,
    neighbors_lbl_indices,
    neighbors_scores,
    self_and_neighbors_attn_mask,
    num_Z,
    model,
    bsz=2000,
):
    """Get Label Representations when we already have embeddings"""
    with evaluating(model), torch.no_grad():
        for i in tqdm(range(0, num_Z, bsz)):
            batch_embeddings = torch.FloatTensor(unnorm_lbl_embeddings[i: i + bsz]).to(model.device)
            batch_neighbors_lbl_indices = torch.LongTensor(neighbors_lbl_indices[i: i + bsz]).to(model.device)
            batch_neighbors_scores = torch.LongTensor(neighbors_scores[i: i + bsz]).to(model.device)
            batch_self_and_neighbors_attn_mask = torch.from_numpy(self_and_neighbors_attn_mask[i: i + bsz]).bool().to(model.device)
            _batch_embeddings = (
                model.encode_label_combined_repr(
                    lbl_embeddings=batch_embeddings,
                    neighbors_index=batch_neighbors_lbl_indices,
                    neighbors_scores=batch_neighbors_scores,
                    self_and_neighbors_attention_mask=batch_self_and_neighbors_attn_mask,
                )
                .cpu()
                .numpy()
            )
            if i == 0:
                embeddings = np.zeros((num_Z, _batch_embeddings.shape[1]))
            embeddings[i: i + batch_embeddings.shape[0]] = _batch_embeddings
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
        print("Will remove instances without any labels during evaluation")
        acc = xc_metrics.Metrics(_true, inv_propen, remove_invalid=True)
        acc = acc.eval(_pred, 10)
    else:
        print("Only R@k is computed. Don't be surprised with 0 val of others")
        acc = np.zeros((4, 10))
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
    print(f"Using filter file for evaluation: {filter_labels}")
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
    clf = torch.from_numpy(clf.astype(np.float32)).to(device).T
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
                     A, B, k=10, mode='ova', huge=False, device="cuda:0"):
    """
    Predict on validation set and evaluate
    * support for filter file (pass "" or empty file otherwise)
    * ova will get top-k predictions but anns would get 300 (change if required)"""
    mode='anns' if huge else mode
    if mode == 'ova':
        pred = _predict_ova(normalize(features, copy=True), normalize(clf, copy=True), k=k, device=device)
    else:
        pred = _predict_anns(features, clf, k=300)
    labels.indices = labels.indices.astype('int64')
    acc, r = evaluate_with_filter(labels, pred, trn_labels, filter_labels, k, A, B, huge)
    return acc, r, pred


def validate_scores_direct(args, snet, val_X_Y, trn_X_Y, eval_neighbors_lbl_indices, eval_neighbors_scores, val_doc_embeddings, lbl_embeddings, prefix=f"", mode="ova", epoch="latest", device="cuda:1"):
    self_and_neighbors_attn_mask = np.zeros(
        (eval_neighbors_lbl_indices.shape[0], args.num_neighbors + 1), dtype=np.int64
    )
    self_and_neighbors_attn_mask[:, 0] = 1
    self_and_neighbors_attn_mask[:, 1:] = eval_neighbors_lbl_indices[:, :args.num_neighbors] < args.num_lbls

    label_representations = get_lbl_representations_zsxc_scores_wo_encoder(
        unnorm_lbl_embeddings=lbl_embeddings,
        neighbors_lbl_indices=eval_neighbors_lbl_indices[:, :args.num_neighbors],
        neighbors_scores=eval_neighbors_scores[:, :args.num_neighbors],
        self_and_neighbors_attn_mask=self_and_neighbors_attn_mask,
        num_Z=val_X_Y.shape[1],
        model=snet,
    )

    np.save(
        os.path.join(args.result_dir, "embeddings", f"{prefix}_{epoch}.zsxc.npy"),
        label_representations,
    )

    if "unseen" in prefix:
        args_filter_labels = args.tst_unseen_filter_labels
    else:
        args_filter_labels = args.tst_seen_filter_labels
    
    if args_filter_labels == "":
        filter_labels = None
    else:
        filter_labels = os.path.join(args.zero_shot_dir, args_filter_labels)
    acc, r, pred = predict_and_eval(
        val_doc_embeddings,
        label_representations,
        val_X_Y,
        trn_X_Y,
        filter_labels,
        A=args.A,
        B=args.B,
        k=args.k,
        mode=mode,
        huge=args.huge,
        device=device,
    )
    del label_representations
    gc.collect()
    return acc, r