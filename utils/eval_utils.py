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
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


@timeit
def get_irene_lbl_representations(
    unnorm_lbl_embeddings,
    neighbor_indices,
    neighbor_scores,
    self_and_neighbors_attn_mask,
    model,
    batch_size=2048,
):
    num_lbls = len(unnorm_lbl_embeddings)
    with evaluating(model), torch.no_grad():
        for i in range(0, num_lbls, batch_size):
            b_lbl_embeddings = torch.FloatTensor(
                unnorm_lbl_embeddings[i : i + batch_size]
            ).to(model.device)
            b_neighbor_indices = torch.LongTensor(
                neighbor_indices[i : i + batch_size]
            ).to(model.device)
            b_neighbor_scores = torch.LongTensor(
                neighbor_scores[i : i + batch_size]
            ).to(model.device)
            b_self_and_neighbors_attn_mask = (
                torch.from_numpy(self_and_neighbors_attn_mask[i : i + batch_size])
                .bool()
                .to(model.device)
            )

            b_irene_reprs = (
                model.encode_label_combined_repr(
                    lbl_embeddings=b_lbl_embeddings,
                    neighbors_index=b_neighbor_indices,
                    neighbors_scores=b_neighbor_scores,
                    self_and_neighbors_attention_mask=b_self_and_neighbors_attn_mask,
                )
                .cpu()
                .numpy()
            )
            if i == 0:
                irene_reprs = np.zeros((num_lbls, b_irene_reprs.shape[1]))
            irene_reprs[i : i + b_lbl_embeddings.shape[0]] = b_irene_reprs
    return irene_reprs


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
    _true.indices = _true.indices.astype("int64")
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


def evaluate_with_filter(
    true_labels, predicted_labels, train_labels, filter_labels, k, A, B, recall_only
):
    """Evaluate function with support of filter file
    * use recall_only = True when dataset is large
    * k: used only for recall@k (precision etc., are computed till @5)
    """
    print(f"Using filter file for evaluation: {filter_labels}")
    mapping = get_filter_map(filter_labels)
    predicted_labels = filter_predictions(predicted_labels, mapping)
    return evaluate(true_labels, predicted_labels, train_labels, k, A, B, recall_only)


def _predict_anns(X, clf, k, M=100, efC=300):
    """
    Train a nearest neighbor structure on label embeddings
    - for a given test point: query the graph for closest label
    - HNSW graph would return cosine distance between and document and
    """
    num_instances, num_labels = len(X), len(clf)
    graph = Shortlist(
        method="hnswlib",
        M=M,
        efC=efC,
        efS=k,
        num_neighbours=k,
        space="cosine",
        num_threads=64,
    )
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
    batches = np.array_split(range(num_instances), num_instances // batch_size)
    output = SMatrix(n_rows=num_instances, n_cols=num_labels, nnz=k)
    X = torch.from_numpy(X)
    clf = torch.from_numpy(clf.astype(np.float32)).to(device).T
    for ind in tqdm(batches):
        s_ind, e_ind = ind[0], ind[-1] + 1
        _X = X[s_ind:e_ind].to(device)
        ans = _X @ clf
        vals, ind = torch.topk(ans, k=k, dim=-1, sorted=True)
        output.update_block(s_ind, ind.cpu().numpy(), vals.cpu().numpy())
        del _X
    if return_sparse:
        return output.data()
    else:
        return output.data("dense")[0]


def predict_and_eval(
    features,
    clf,
    labels,
    trn_labels,
    filter_labels,
    A=0.55,
    B=1.5,
    k=10,
    mode="ova",
    huge=False,
    device="cuda:0",
):
    """
    Predict on validation set and evaluate
    * support for filter file (pass "" or empty file otherwise)
    * ova will get top-k predictions but anns would get 300 (change if required)"""
    mode = "anns" if huge else mode
    if mode == "ova":
        pred = _predict_ova(features.copy(), clf.copy(), k=k, device=device)
    else:
        pred = _predict_anns(features, clf, k=300)
    labels.indices = labels.indices.astype("int64")
    acc, r = evaluate_with_filter(
        labels, pred, trn_labels, filter_labels, k, A, B, huge
    )
    return acc, r, pred


def validate(
    args,
    net,
    Y_eval,
    eval_neighbor_indices,
    eval_neighbor_scores,
    eval_doc_embeddings,
    lbl_embeddings,
    prefix=f"",
    mode="ova",
    epoch="latest",
    device="cuda:1",
):
    self_and_neighbors_attn_mask = np.zeros(
        (eval_neighbor_indices.shape[0], args.num_neighbors + 1), dtype=np.int64
    )
    self_and_neighbors_attn_mask[:, 0] = 1
    self_and_neighbors_attn_mask[:, 1:] = (
        eval_neighbor_indices[:, : args.num_neighbors] < args.num_trn_lbls
    )

    irene_lbl_representations = get_irene_lbl_representations(
        unnorm_lbl_embeddings=lbl_embeddings,
        neighbor_indices=eval_neighbor_indices[:, : args.num_neighbors],
        neighbor_scores=eval_neighbor_scores[:, : args.num_neighbors],
        self_and_neighbors_attn_mask=self_and_neighbors_attn_mask,
        model=net,
    )

    os.makedirs(os.path.join(args.OUT_DIR, "embeddings"), exist_ok=True)
    np.save(
        os.path.join(args.OUT_DIR, "embeddings", f"Y_{prefix}_{epoch}.irene.npy"),
        irene_lbl_representations,
    )

    if "zero" in prefix:
        filter_labels = args.filter_labels_zero_filename
    else:
        filter_labels = args.filter_labels_full_filename

    if filter_labels == "":
        filter_labels = None
    else:
        filter_labels = os.path.join(args.DATA_DIR, filter_labels)

    acc, r, pred = predict_and_eval(
        features=eval_doc_embeddings,
        clf=irene_lbl_representations,
        labels=Y_eval,
        trn_labels=csr_matrix((len(eval_doc_embeddings), len(irene_lbl_representations))),
        filter_labels=filter_labels,
        k=args.eval_k,
        mode=args.eval_mode,
        device=device,
    )
    del irene_lbl_representations
    gc.collect()
    return acc, r
