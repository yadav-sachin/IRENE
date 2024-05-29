# Standard library imports
import argparse
import functools
import gc
import inspect
import math
import os
import pickle
import sys
import time
import warnings

# Third-party imports
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
import wandb
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
from xclib.data import data_utils
import xclib.evaluation.xc_metrics as xc_metrics
from xclib.utils.clustering import cluster_balance, b_kmeans_dense
from xclib.utils.matrix import SMatrix
from xclib.utils.shortlist import Shortlist
from xclib.utils.sparse import csr_from_arrays

# Local application/library specific imports
from cluster_gpu import balanced_cluster
from data import DatasetDNeighbors, collate_fn_neighbors
from modeling import (
    timeit,
    validate_seen_clf_only_direct,
    validate_scores_direct,
)
from modeling import CustomEmbedding
from modeling import TripletMarginLossOHNM

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


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pos_embeddings = torch.nn.Parameter(torch.zeros(2, d_model))

    def forward(self, x):
        # x of shape (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        position_ids = torch.zeros(seq_len).long()
        position_ids[1:] = 1
        position_ids = position_ids.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        position_embeddings = self.pos_embeddings[position_ids]
        return x + position_embeddings


class ScoreEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score_embeddings = torch.nn.Parameter(torch.zeros(11, d_model))

    # x is of shape (bs, seq_len, d_model), scores is of shape (bs, seq_len)
    def forward(self, x, scores):
        score_embeddings = self.score_embeddings[scores]
        return x + score_embeddings

class SiameseNetworkZSXC(torch.nn.Module):
    """
    A network class to support Siamese style training
    * specialized for sentence-bert or hugging face
    * hard-coded to use a joint encoder

    """

    def __init__(
        self,
        transform_dim,
        n_neighbors,
        n_lbls,
        n_heads,
        n_encoder_layers,
        dropout,
        device,
    ):
        super(SiameseNetworkZSXC, self).__init__()
        self.device = device
        self.n_lbls = n_lbls
        self.n_neighbors = n_neighbors
        self.dropout = dropout

        self.dim = transform_dim
        if self.dim < 0:
            self.dim = args.dim

        self.classifiers = torch.nn.Parameter(torch.Tensor(n_lbls + 1, self.dim))
        self.seen_lbl_embeddings = torch.nn.Parameter(torch.Tensor(n_lbls + 1, self.dim))

        self.doc_drop_transform = torch.nn.Dropout(p=0.5)

        torch.nn.init.xavier_uniform_(self.classifiers)
        self.pos_embedd = PositionalEmbedding(d_model=self.dim)
        self.score_embedd = ScoreEmbedding(d_model=self.dim)

        with torch.no_grad():
            self.classifiers[self.n_lbls] = 0
            self.classifiers[self.n_lbls].requires_grad = False
            self.seen_lbl_embeddings[self.n_lbls] = 0
            self.seen_lbl_embeddings[self.n_lbls].requires_grad = False

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=n_heads, batch_first=True, dropout=dropout
        )
        self.combiner = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

    def label_classifier_repr(
        self,
    ):
        return (self.classifiers[:-1])

    def encode_label_combined_repr(
        self,
        lbl_embeddings,
        neighbors_index,
        neighbors_scores,
        self_and_neighbors_attention_mask,
        lbl_index=None,
    ):
        neighbors_scores = neighbors_scores.to(self.device)
        neighbors_index, self_and_neighbors_attention_mask = neighbors_index.to(self.device), self_and_neighbors_attention_mask.to(self.device)

        if lbl_index is not None:
            lbl_index = lbl_index.to(self.device)
            lbl_embedd = F.embedding(lbl_index.squeeze(1), self.seen_lbl_embeddings)
        else:
            lbl_embedd = lbl_embeddings.to(self.device)

        neighbors_clfs = F.embedding(neighbors_index, self.classifiers)
            
        # Score Embedding is added here
        neighbors_clfs = self.score_embedd(neighbors_clfs, neighbors_scores)
        combined_reprs = torch.cat([lbl_embedd.unsqueeze(1), neighbors_clfs], dim=1)

        # Positional Embedding is added here
        combined_reprs = self.pos_embedd(combined_reprs)

        combined_reprs = self.combiner(
            combined_reprs,
            src_key_padding_mask=~(self_and_neighbors_attention_mask.bool()),
        )
        combined_reprs = (
            torch.sum(
                combined_reprs * self_and_neighbors_attention_mask.unsqueeze(2)[:, :combined_reprs.shape[1]], dim=1
            )
            / torch.sum(self_and_neighbors_attention_mask, dim=1)[:, None]
        )
        return (combined_reprs)

    def forward(
        self,
        doc_embeddings=None,
        lbl_embeddings=None,
        neighbors_index=None,
        neighbors_scores=None,
        self_and_neighbors_attention_mask=None,
        lbl_index=None,
    ):
        if doc_embeddings is None:
            return self.encode_label_combined_repr(
                lbl_embeddings=lbl_embeddings,
                neighbors_index=neighbors_index,
                neighbors_scores=neighbors_scores,
                self_and_neighbors_attention_mask=self_and_neighbors_attention_mask,
                lbl_index=lbl_index,
            )
        elif lbl_embeddings is None and lbl_index is None:
            return doc_embeddings
        label_reprs = self.encode_label_combined_repr(
            lbl_embeddings=lbl_embeddings,
            neighbors_index=neighbors_index,
            neighbors_scores=neighbors_scores,
            self_and_neighbors_attention_mask=self_and_neighbors_attention_mask,
            lbl_index=lbl_index,
        )
        if doc_embeddings is not None:
            return doc_embeddings, label_reprs
        return label_reprs

    @property
    def repr_dims(self):
        return args.dim


class MySampler(torch.utils.data.Sampler[int]):
    def __init__(self, order):
        self.order = order.copy()

    def update_order(self, x):
        self.order[:] = x[:]

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)

def prepare_loss(args, margin, num_negatives):
    """
    Set-up the loss function
    * num_violators can be printed, if required *
    """
    criterion = nn.BCEWithLogitsLoss()
    return criterion


def prepare_network(args):
    """
    Set-up the network

    * Use DP if multiple GPUs are available
    """
    print("==> Creating model, optimizer...")

    snet = SiameseNetworkZSXC(
        transform_dim=args.transform_dim,
        n_neighbors=args.num_neighbors,
        n_lbls=args.num_lbls,
        n_heads=args.num_heads,
        n_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        device=args.device,
    )
    snet.to(args.device)

    print(snet)
    return snet


def prepare_optimizer_and_schedular(args, snet, len_train_loader):
    """
    Set-up the optimizer and schedular

    * t_total has to be pre-calculated (lr will be zero after these many steps)
    """
    no_decay = ["bias", "LayerNorm.weight"]
    clf_gp = [
        {"params": snet.classifiers, "weight_decay": 0.0},
    ]
    combiner_gp = [
        {
            "params": [
                p
                for n, p in snet.combiner.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in snet.combiner.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {"params": snet.pos_embedd.pos_embeddings, "weight_decay": 0.0},
        {"params": snet.score_embedd.score_embeddings, "weight_decay": 0.0},
    ]
    clf_optimizer = transformers.AdamW(clf_gp, **{"lr": args.clf_lr, "eps": 1e-08})
    combiner_optimizer = transformers.AdamW(combiner_gp, **{"lr": args.lr, "eps": 1e-06})
    
    clf_scheduler = transformers.get_linear_schedule_with_warmup(
        clf_optimizer, num_warmup_steps=500, num_training_steps=args.epochs * len_train_loader
    )
    combiner_scheduler = transformers.get_linear_schedule_with_warmup(
        combiner_optimizer, num_warmup_steps=500, num_training_steps=(args.epochs - args.combiner_start_epoch) * len_train_loader
    )
    return clf_optimizer, combiner_optimizer, clf_scheduler, combiner_scheduler


def ova_prediction(data, classifiers, top_k=50, batch_size=256, device="cuda"):
    torch.set_grad_enabled(False)
    num_instances = len(data)
    num_labels = len(classifiers)
    batches = np.array_split(np.arange(num_instances), num_instances // batch_size)
    predictions = SMatrix(n_rows=num_instances, n_cols=num_labels, nnz=top_k)
    data = torch.from_numpy(data)
    classifiers = torch.from_numpy(classifiers).to(device).T
    offset = 0
    for batch in tqdm(batches):
        start, end = batch[0], batch[-1] + 1
        temp = data[start:end].to(device)
        logits = temp @ classifiers
        vals, ind = torch.topk(logits, k=top_k, dim=-1, sorted=True)
        predictions.update_block(offset, ind.cpu().numpy(), vals.cpu().numpy())
        offset += len(batch)
        del temp
    del classifiers
    del data
    torch.cuda.empty_cache()
    return predictions.data()

def prepare_data(args, encoder_trn_doc_embedds):
    print("==> Creating Dataloader...")
    print("Using Doc Side Sampling")
    train_dataset = DatasetDNeighbors(
        encoder_trn_doc_embedds=encoder_trn_doc_embedds,
        seen_trn_X_Y_fname=os.path.join(args.zero_shot_dir, args.trn_lbl_fname),
        seen_neighbors_lbl_indices=args.seen_neighbors_lbl_indices,
        seen_neighbors_scores=args.seen_neighbors_scores,
        num_seen_lbls=args.num_lbls,
        num_neighbors=args.num_neighbors,
    )
    
    train_order = np.random.permutation(len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=15,
        prefetch_factor=7,
        collate_fn=partial(collate_fn_neighbors),
        batch_sampler=torch.utils.data.sampler.BatchSampler(
            MySampler(train_order), args.batch_size, False
        ),
    )
    return train_loader

@timeit
def cluster_items_gpu(X, tree_depth, args):
    clusters = balanced_cluster(
        embs=torch.HalfTensor(X), 
        num_levels=tree_depth, 
        devices=args.cls_devices, 
        verbose=True)
    clustering_mat = csr_matrix(
        (
            np.ones(sum([len(c) for c in clusters])),
            np.concatenate(clusters),
            np.cumsum([0, *[len(c) for c in clusters]]),
        ),
        shape=(len(clusters), X.shape[0]),
    )
    return clustering_mat

@timeit
def cluster_items(X, depth, n_threads):
    n_clusters = 2 ** (depth - 1)
    clusters, _ = cluster_balance(
        X=X.copy(),
        clusters=[np.arange(len(X), dtype=np.int)],
        num_clusters=n_clusters,
        splitter=b_kmeans_dense,
        num_threads=n_threads,
        verbose=True,
    )
    clustering_mat = csr_matrix(
        (
            np.ones(sum([len(c) for c in clusters])),
            np.concatenate(clusters),
            np.cumsum([0, *[len(c) for c in clusters]]),
        ),
        shape=(len(clusters), X.shape[0]),
    )
    return clustering_mat


def log_precision_recall_metrics_wandb(acc, r, prefix, epoch):
    wandb.log(
        {
            "epoch": epoch,
            f"{prefix}_Recall@1": r[0],
            f"{prefix}_Recall@3": r[2],
            f"{prefix}_Recall@5": r[4],
            f"{prefix}_Recall@10": r[9],
            f"{prefix}_Recall@30": r[29],
            f"{prefix}_Recall@50": r[49],
            f"{prefix}_Recall@100": r[99],
        }
    )
    wandb.log(
        {
            "epoch": epoch,
            f"{prefix}_Precision@1": acc[0][0],
            f"{prefix}_Precision@3": acc[0][2],
            f"{prefix}_Precision@5": acc[0][4],
            f"{prefix}_PSP@1": acc[2][0],
            f"{prefix}_PSP@3": acc[2][2],
            f"{prefix}_PSP@5": acc[2][4],
        }
    )
    print("Epoch: ", epoch, f"{prefix}_Precision(1,3,5) :", acc[0][0], acc[0][2], acc[0][4])
    print("Epoch:", epoch, f"{prefix}_Recall(1,3,5,10,30,50,100) :", r[0], r[2], r[4], r[9], r[29], r[99])
    print("Epoch", epoch, f"{prefix}_PSP(1,3,5) :", acc[2][0], acc[2][2], acc[2][4])


def zero_shot_one_shot_evaluation(args, snet, epoch, device):
    """
    Evaluates the zero-shot performance.
    Args:
    args: Argument object containing necessary parameters.
    snet: The neural network model.
    epoch: Current epoch number.
    device: Device to run the model on.
    """
    print("Generating Unseen Neighbors Based on Classifiers")
    seen_label_norm_clfs = (snet.classifiers.data.detach()[:-1]).cpu().numpy()
    # Get these two things from the arguments
    unseen_lbl_embedd_neighbors_indices = args.unseen_encoder_lbl_indices.copy()
    norm_unseen_lbl_embeddings = args.unnorm_unseen_lbl_embeddings
    unseen_one_shot_trn_doc_embeddings = args.unseen_lbl_one_doc_embeddings.copy()

    unseen_lbl_tail_0_neighbors_dots = unseen_lbl_embedd_neighbors_indices.copy().astype(np.float64)
    unseen_lbl_tail_1_neighbors_dots = unseen_lbl_embedd_neighbors_indices.copy().astype(np.float64)

    for unseen_lbl_id in tqdm(range(len(unseen_lbl_embedd_neighbors_indices))):
        curr_unseen_embedding = norm_unseen_lbl_embeddings[unseen_lbl_id]
        curr_trn_doc_embedding = unseen_one_shot_trn_doc_embeddings[unseen_lbl_id]

        seen_neighbors_clfs = seen_label_norm_clfs[unseen_lbl_embedd_neighbors_indices[unseen_lbl_id]]

        unseen_lbl_tail_0_neighbors_dots[unseen_lbl_id] = (seen_neighbors_clfs.dot(curr_unseen_embedding))
        unseen_lbl_tail_1_neighbors_dots[unseen_lbl_id] = (seen_neighbors_clfs.dot(curr_trn_doc_embedding))

    unseen_lbl_tail_0_neighbors_indices_thresholds = []
    unseen_lbl_tail_0_neighbors_scores_thresholds = []

    unseen_lbl_tail_1_neighbors_indices_thresholds = []
    unseen_lbl_tail_1_neighbors_scores_thresholds = []

    for threshold in tqdm(range(0, 70, 5)):
        clf_threshold = threshold / 100
        
        unseen_lbl_tail_0_neighbors_scores_tmp = unseen_lbl_embedd_neighbors_indices.copy()
        unseen_lbl_tail_0_neighbors_indices_tmp = unseen_lbl_embedd_neighbors_indices.copy()

        for unseen_lbl_id in range(len(unseen_lbl_tail_0_neighbors_scores_tmp)):
            doc_intersect_freq = unseen_lbl_tail_0_neighbors_dots[unseen_lbl_id].copy()
            doc_intersect_freq[doc_intersect_freq < clf_threshold] = 0
            doc_intersect_freq[doc_intersect_freq >= clf_threshold] = 1
            correct_neigh_order = np.argsort(-1 * doc_intersect_freq, kind="stable")

            unseen_lbl_tail_0_neighbors_indices_tmp[unseen_lbl_id] = unseen_lbl_embedd_neighbors_indices[unseen_lbl_id][correct_neigh_order]
            unseen_lbl_tail_0_neighbors_scores_tmp[unseen_lbl_id] = doc_intersect_freq[correct_neigh_order]

        unseen_lbl_tail_0_neighbors_scores_tmp = np.around(unseen_lbl_tail_0_neighbors_scores_tmp).astype(np.int32)
        unseen_lbl_tail_0_neighbors_indices_tmp[unseen_lbl_tail_0_neighbors_scores_tmp == 0] = args.num_lbls

        unseen_lbl_tail_0_neighbors_indices_thresholds.append(unseen_lbl_tail_0_neighbors_indices_tmp)
        unseen_lbl_tail_0_neighbors_scores_thresholds.append(unseen_lbl_tail_0_neighbors_scores_tmp)

    for threshold in tqdm(range(0, 70, 5)):
        clf_threshold = threshold / 100

        unseen_lbl_tail_1_neighbors_scores_tmp = unseen_lbl_embedd_neighbors_indices.copy()
        unseen_lbl_tail_1_neighbors_indices_tmp = unseen_lbl_embedd_neighbors_indices.copy()

        for unseen_lbl_id in range(len(unseen_lbl_tail_1_neighbors_scores_tmp)):
            # Max-Voting 
            # Voting (from unseen label embeddings)
            doc_intersect_freq = unseen_lbl_tail_0_neighbors_dots[unseen_lbl_id].copy()
            doc_intersect_freq[doc_intersect_freq < clf_threshold] = 0
            doc_intersect_freq[doc_intersect_freq >= clf_threshold] = 1
            # Voting from given train doc embeddings
            doc_intersect_freq += (unseen_lbl_tail_1_neighbors_dots[unseen_lbl_id] >= clf_threshold).astype(np.int32)
            correct_neigh_order = np.argsort(-1 * doc_intersect_freq, kind="stable")

            unseen_lbl_tail_1_neighbors_indices_tmp[unseen_lbl_id] = unseen_lbl_embedd_neighbors_indices[unseen_lbl_id][correct_neigh_order]
            unseen_lbl_tail_1_neighbors_scores_tmp[unseen_lbl_id] = doc_intersect_freq[correct_neigh_order]

        unseen_lbl_tail_1_neighbors_scores_tmp = np.around(unseen_lbl_tail_1_neighbors_scores_tmp).astype(np.int32)
        unseen_lbl_tail_1_neighbors_indices_tmp[unseen_lbl_tail_1_neighbors_scores_tmp == 0] = args.num_lbls

        unseen_lbl_tail_1_neighbors_indices_thresholds.append(unseen_lbl_tail_1_neighbors_indices_tmp)
        unseen_lbl_tail_1_neighbors_scores_thresholds.append(unseen_lbl_tail_1_neighbors_scores_tmp)

    # Start Evaluation
    for arr_indx, threshold in tqdm(enumerate(range(0, 70, 5))):
        acc, r = validate_scores_direct(
            args=args,
            snet=snet,
            val_X_Y=args.unseen_tst_X_Y,
            trn_X_Y=args.unseen_trn_X_Y,
            eval_neighbors_lbl_indices=unseen_lbl_tail_0_neighbors_indices_thresholds[arr_indx],
            eval_neighbors_scores=unseen_lbl_tail_0_neighbors_scores_thresholds[arr_indx],
            val_doc_embeddings=args.encoder_unseen_tst_doc_embedd,
            lbl_embeddings=args.unnorm_unseen_lbl_embeddings,
            prefix="unseen_lbl",
            epoch=epoch,
            mode=args.pred_mode,
            device=device,
        )
        log_precision_recall_metrics_wandb(acc=acc, r=r, prefix=f"unseen_0_lbl_thres_{threshold}", epoch=epoch)

    for arr_indx, threshold in tqdm(enumerate(range(0, 70, 5))):
        acc, r = validate_scores_direct(
            args=args,
            snet=snet,
            val_X_Y=args.unseen_tst_X_Y,
            trn_X_Y=args.unseen_trn_X_Y,
            eval_neighbors_lbl_indices=unseen_lbl_tail_1_neighbors_indices_thresholds[arr_indx],
            eval_neighbors_scores=unseen_lbl_tail_1_neighbors_scores_thresholds[arr_indx],
            val_doc_embeddings=args.encoder_unseen_tst_doc_embedd,
            lbl_embeddings=args.unnorm_unseen_lbl_embeddings,
            prefix="unseen_lbl",
            epoch=epoch,
            mode=args.pred_mode,
            device=device,
        )
        log_precision_recall_metrics_wandb(acc=acc, r=r, prefix=f"unseen_1_lbl_thres_{threshold}", epoch=epoch)


def validate_metrics(args, snet, epoch):
    device="cuda:1"
    if torch.cuda.device_count() <= 1:
        device="cuda:0"
    with evaluating(snet), torch.no_grad():
        # acc, r = validate_unseen_encoder_only(
        #     args,
        #     snet,
        #     args.unseen_neighbors_lbl_indices,
        #     epoch=epoch,
        #     mode=args.pred_mode,
        #     device=device,
        # )
        # log_precision_recall_metrics_wandb(acc=acc, r=r, prefix="encoder_unseen", epoch=epoch)
        # acc, r = validate_seen_encoder_only(
        #     args,
        #     snet,
        #     args.seen_neighbors_lbl_indices,
        #     epoch=epoch,
        #     mode=args.pred_mode,
        #     device=device,
        # )
        # log_precision_recall_metrics_wandb(acc=acc, r=r, prefix="encoder_seen", epoch=epoch)
        acc, r = validate_seen_clf_only_direct(
            args=args,
            snet=snet,
            val_X_Y=args.seen_tst_X_Y,
            trn_X_Y=args.seen_trn_X_Y,
            val_doc_embeddings=args.encoder_seen_tst_doc_embedd,
            mode=args.pred_mode,
            epoch=epoch,
            device=device,
        )
        log_precision_recall_metrics_wandb(acc=acc, r=r, prefix="classifier_seen", epoch=epoch)
        if epoch >= args.combiner_start_epoch: 
            zero_shot_one_shot_evaluation(args, snet, epoch, device)

            acc, r = validate_scores_direct(
                args=args,
                snet=snet,
                val_X_Y=args.seen_tst_X_Y,
                trn_X_Y=args.seen_trn_X_Y,
                eval_neighbors_lbl_indices=args.seen_neighbors_lbl_indices,
                eval_neighbors_scores=args.seen_neighbors_scores,
                val_doc_embeddings=args.encoder_seen_tst_doc_embedd,
                lbl_embeddings=args.unnorm_seen_lbl_embeddings,
                prefix="seen_lbl",
                epoch=epoch,
                mode=args.pred_mode,
                device=device,
            )
            log_precision_recall_metrics_wandb(acc=acc, r=r, prefix="seen", epoch=epoch)


def train(args, snet, criterion_combiner, criterion_clf, clf_optimizer, clf_scheduler, combiner_optimizer, combiner_scheduler, train_loader, trn_doc_embedds):

    with evaluating(snet), torch.no_grad():
        args.encoder_seen_tst_doc_embedd = args.seen_tst_doc_embeddings
        print("Loaded Seen Test Doc Embeddings")

    with evaluating(snet), torch.no_grad():
        args.encoder_unseen_tst_doc_embedd = args.unseen_tst_doc_embeddings
        print("Loaded Unseen Test Doc Embeddings")


    start_time = time.time()
    val_time = 0
    n_iter = 0

    validate_metrics(args=args, snet=snet, epoch=-1)
    if args.save_model:
        snet.eval()
        state_dict = {}
        for k, v in snet.state_dict().items():
            state_dict[k.replace("module.", "")] = v
        torch.save(state_dict, f"{args.model_dir}/state_dict_-1.pt")
        with open(f"{args.model_dir}/executed_args.pkl", "wb") as fout:
            pickle.dump(args, fout)
        with open(f"{args.model_dir}/executed_script.py", "w") as fout:
            print(inspect.getsource(sys.modules[__name__]), file=fout)
        with open(f"{args.model_dir}/executed_script_args.txt", "w") as fout:
            print(args, file=fout)

    total_steps = len(train_loader) * args.epochs

    args.cl_size = 1

    for epoch in range(args.epochs):
        snet.train()
        torch.set_grad_enabled(True)
        pbar = tqdm(train_loader)
        vio1_history, vio2_history = [], []
        loss1_history, loss2_history, loss_history = [], [], []
        t1 = time.time()
        for data in pbar:
            snet.zero_grad()
            batch_size = data["batch_size"]

            lbl_classifiers = snet.classifiers[data["lbl_indices"]].squeeze(1)
            ip_embeddings = data["doc_embeddings"].to(snet.device)
            # Same dropout for documents
            dropout_ip_embeddings = snet.doc_drop_transform(ip_embeddings)
            loss2 = criterion_clf(
                dropout_ip_embeddings @ lbl_classifiers.T, data["Y"].float().to(args.device)
            )

            # avg_violators2 = torch.mean(violators2 / args.num_negatives).item() * 100
            # vio2_history += violators2.tolist()
            
            if epoch >= args.combiner_start_epoch:
                op_embeddings = snet.forward(
                    neighbors_index=data["neighbors_indices"],
                    neighbors_scores=data["neighbors_scores"],
                    self_and_neighbors_attention_mask=data["self_and_neighbors_mask"],
                    lbl_index=data["lbl_indices"],
                )

                loss1 = criterion_combiner(
                    ip_embeddings @ op_embeddings.T, data["Y"].float().to(args.device)
                )
                # avg_violators1 = torch.mean(violators1 / args.num_negatives).item() * 100
                # vio1_history += violators1.tolist()

                loss = loss1 + loss2

                loss1_history.append(loss1.item())
                loss2_history.append(loss2.item())
                loss_history.append(loss.item())

                loss.backward()
                clf_optimizer.step()
                combiner_optimizer.step()

                clf_scheduler.step()
                combiner_scheduler.step()
            else:
                loss = loss2
                loss.backward()
                clf_optimizer.step()
                clf_scheduler.step()

                loss2_history.append(loss2.item())
                loss_history.append(loss.item())

            pbar.set_description(
                "epoch: {}, loss: {:4e}".format(
                    epoch, loss.item()
                )
            )

            n_iter += 1

        if args.num_violators:
            # epoch_avg_violators1 = (
            #     np.mean(np.array(vio1_history) / args.num_negatives) * 100
            # )
            # epoch_avg_violators2 = (
            #     np.mean(np.array(vio2_history) / (args.num_negatives * args.num_neighbors) ) * 100
            # )
            # print(
            #     f"epoch {epoch} time taken {time.time() - t1} || average loss = {'{:.5e}'.format(np.mean(loss_history))}, average violators(%)={'{:.2f}'.format(epoch_avg_violators1)}"
            # )
            # wandb.log(
            #     {
            #         "loss1": np.mean(loss1_history),
            #         # "violators1(%)": epoch_avg_violators1,
            #         "epoch": epoch,
            #     }
            # )
            wandb.log(
                {
                    "clf_lr": clf_scheduler.get_last_lr()[0],
                    "combiner_lr": combiner_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
            )
            if epoch >= args.combiner_start_epoch:
                # epoch_avg_violators1 = (
                #     np.mean(np.array(vio1_history) / (args.num_negatives * args.num_neighbors) ) * 100
                # )
                wandb.log(
                    {
                        "loss1": np.mean(loss1_history),
                        # "violators1(%)": epoch_avg_violators1,
                        "epoch": epoch,
                    }
                )

            wandb.log(
                {
                    "loss2": np.mean(loss2_history),
                    # "violators2(%)": epoch_avg_violators2,
                    "epoch": epoch,
                }
            )
            wandb.log(
                {
                    "loss": np.mean(loss_history),
                    "epoch": epoch,
                }
            )
        else:
            print(
                f"epoch {epoch} time taken {time.time() - t1} || average loss = {'{:.5e}'.format(np.mean(loss_history))}"
            )
            wandb.log({"loss": np.mean(loss_history), "epoch": epoch})

        if epoch >= args.clf_cl_start:
            if epoch == args.clf_cl_start:
                args.cl_size = args.clf_cl_size
            if epoch == args.combiner_cl_start:
                args.cl_size = args.combiner_cl_size
            if epoch == args.combiner_start_epoch - 1:
                args.cl_size = 1
            if ((epoch - args.clf_cl_start) % args.cl_update == 0) or (epoch == args.clf_cl_start) or (epoch == args.combiner_cl_start) or (epoch == args.combiner_start_epoch - 1):
                print(
                    f"Updating clusters with cluster size {args.cl_size} (using stale embeddings)"
                )
                embs = trn_doc_embedds.copy()
                tree_depth = (
                    int(np.ceil(np.log(embs.shape[0] / args.cl_size) / np.log(2)))
                )
                print(f"tree depth = {tree_depth}")
                if args.cls_devices is None:
                    cluster_mat = cluster_items(embs, tree_depth, 16).tocsr()
                else:
                    cluster_mat = cluster_items_gpu(embs, tree_depth, args).tocsr()
                del embs
                gc.collect()

            print("Updating train order...")
            cmat = cluster_mat[np.random.permutation(cluster_mat.shape[0])]
            train_loader.batch_sampler.sampler.update_order(cmat.indices)
        else:
            train_loader.batch_sampler.sampler.update_order(
                np.random.permutation(len(train_loader.dataset))
            )

        if epoch in args.curr_steps:
            args.cl_size *= 2
            print(f"Changing cluster size to {args.cl_size}")


        if epoch in args.eval_epochs:
            validate_metrics(args=args, snet=snet, epoch=epoch)
            if args.save_model:
                snet.eval()
                state_dict = {}
                for k, v in snet.state_dict().items():
                    state_dict[k.replace("module.", "")] = v
                torch.save(state_dict, f"{args.model_dir}/state_dict_{epoch}.pt")
                with open(f"{args.model_dir}/executed_script.py", "w") as fout:
                    print(inspect.getsource(sys.modules[__name__]), file=fout)
                with open(f"{args.model_dir}/executed_script_args.txt", "w") as fout:
                    print(args, file=fout)
            

    total_time = time.time() - start_time


def initialize_encoder_classifiers(args, snet):
    # Initialize Seen Label Embeddings
    with evaluating(snet), torch.no_grad():
        snet.seen_lbl_embeddings[:-1].data.copy_(torch.from_numpy(args.unnorm_seen_lbl_embeddings))
        print(f"Loaded Seen Label Embeddings")

    with torch.no_grad():
        snet.seen_lbl_embeddings.requires_grad = False

    # Load Classifiers as Label Embeddings
    with evaluating(snet), torch.no_grad():
        if args.init_classifiers:
            # Initialize Classifiers from un-normalized label embeddings
            snet.classifiers[:-1].data.copy_(torch.from_numpy(args.unnorm_seen_lbl_embeddings))
            print(f"Loaded Classifiers Weights from Un-normalized Seen Label Embeddings")
        if args.init_classifiers_path != "":
            clf_weights = np.load(args.init_classifiers_path)
            snet.classifiers[:-1].data.copy_(torch.from_numpy(clf_weights))
            print(f"Loaded Classifiers Weights from Path: {args.init_classifiers_path}")

    if args.freeze_classifiers:
        print("Freezing Classifiers")
        snet.classifiers.requires_grad = False

    return snet


def main(args):
    wandb.init(
        project=args.wandb_project_name,
        entity="anirudhb",
        name=args.version,
        save_code=True,
        settings=wandb.Settings(code_dir="."),
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "clf_learning_rate": args.clf_lr,
            "batch_size": args.batch_size,
            "margin": args.margin,
            "clf_margin": args.clf_margin,
            "max_length": args.max_length,
            "num_negatives": args.num_negatives,
            "clf_num_negatives": args.clf_num_negatives,
            "transform_dim": args.transform_dim,
            "combiner_num_clusters": args.combiner_num_clusters,
            "cluster_update": args.cl_update,
            "num_lbls": args.num_lbls,
            "num_unseen_lbls": args.num_unseen_lbls,
            "init_classifiers": args.init_classifiers,
            "neighbor_itself": args.neighbor_itself,
            "freeze_classifiers": args.freeze_classifiers,
            "num_neighbors": args.num_neighbors,
            "num_heads": args.num_heads,
            "num_encoder_layers": args.num_encoder_layers,
            "dropout": args.dropout,
            "num_gpus": torch.cuda.device_count(),
            "unnorm_seen_lbl_embeddings": args.unnorm_seen_lbl_embeddings_filepath,
            "unnorm_unseen_lbl_embeddings": args.unnorm_unseen_lbl_embeddings_filepath,
            "seen_neighbors_filepath": args.seen_neighbors_filepath,
            "seen_neighbors_scores_filepath": args.seen_neighbors_scores_filepath,
            "unseen_encoder_neighbors_filepath": args.unseen_encoder_neighbors_filepath,
            "unseen_lbl_one_doc_embeddings": args.unseen_lbl_one_doc_embeddings_filepath,
            "combiner_start_epoch": args.combiner_start_epoch,
            "base_encoder": args.base_encoder,
        },
    )

    args.device = torch.device(args.device)

    args.zero_shot_dir = os.path.join(args.work_dir, "zero_shot_data", args.dataset)

    args.model_dir = os.path.join(
        args.work_dir, "models", "ZSXC_TOP", args.dataset, args.base_encoder, args.version
    )
    args.result_dir = os.path.join(
        args.work_dir, "results", "ZSXC_TOP", args.dataset, args.base_encoder, args.version
    )
    args.data_dir = os.path.join(args.work_dir, "zero_shot_data", args.dataset)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "embeddings"), exist_ok=True)

    # Get GPU clustering devices
    if args.cls_devices == "":
        args.cls_devices = None
    else:
        args.cls_devices = list(map(int, args.cls_devices.split(",")))

    if args.eval_epochs == "":
        args.eval_epochs = None
    else:
        args.eval_epochs = set(list(map(int, args.eval_epochs.split(","))))

        # get curruculum steps
    if args.curr_steps == "":
        args.curr_steps = set()
    else:
        args.curr_steps = set(map(int, args.curr_steps.split(",")))

    # Read seen_tst_X_Y and unseen tst_X_Y
    args.unseen_tst_X_Y = data_utils.read_gen_sparse(
        os.path.join(args.zero_shot_dir, "unseen_tst_X_Y.txt")
    )
    args.unseen_trn_X_Y = data_utils.read_gen_sparse(
        os.path.join(args.zero_shot_dir, "unseen_trn_X_Y.txt")
    ) # This is an empty place-holder file

    args.seen_tst_X_Y = data_utils.read_gen_sparse(
        os.path.join(args.zero_shot_dir, "seen_tst_X_Y.txt")
    )

    args.seen_trn_X_Y = data_utils.read_gen_sparse(
        os.path.join(args.zero_shot_dir, "trn_X_Y.txt")
    )

    criterion_combiner = prepare_loss(args, margin=args.margin, num_negatives=args.num_negatives)
    criterion_clf = prepare_loss(args, margin=args.clf_margin, num_negatives=args.clf_num_negatives)

    args.seen_neighbors_lbl_indices, args.seen_neighbors_scores = np.load(args.seen_neighbors_filepath)[:, :args.num_neighbors], np.load(args.seen_neighbors_scores_filepath)[:, :args.num_neighbors]

    if not args.neighbor_itself:
        args.seen_neighbors_lbl_indices = np.load(args.seen_neighbors_filepath)[:, 1:args.num_neighbors + 1]
        args.seen_neighbors_scores = np.load(args.seen_neighbors_scores_filepath)[:, 1:args.num_neighbors + 1]
    print("Loaded Seen Neighbors")

    args.seen_neighbors_scores[args.seen_neighbors_scores >= 10] = 10
    args.seen_neighbors_lbl_indices[args.seen_neighbors_scores <= 0] = args.num_lbls

    args.unseen_encoder_lbl_indices = np.load(args.unseen_encoder_neighbors_filepath)
    print("Loaded Unseen Neighbors")

    args.trn_doc_embeddings = np.load(args.trn_doc_embeddings_filepath).astype(np.float32)
    print("Loaded Train Doc Embeddings")
    args.seen_tst_doc_embeddings = np.load(args.seen_tst_doc_embeddings_filepath).astype(np.float32)
    print("Loaded Seen Test Doc Embeddings")
    args.unseen_tst_doc_embeddings = np.load(args.unseen_tst_doc_embeddings_filepath).astype(np.float32)
    print("Loaded Unseen Test Doc Embeddings")

    args.unnorm_seen_lbl_embeddings = np.load(args.unnorm_seen_lbl_embeddings_filepath).astype(np.float32)
    args.unnorm_unseen_lbl_embeddings = np.load(args.unnorm_unseen_lbl_embeddings_filepath).astype(np.float32)
    print("Loaded Unnormalized Label Embeddings")

    args.unseen_lbl_one_doc_embeddings = np.load(args.unseen_lbl_one_doc_embeddings_filepath).astype(np.float32)

    snet = prepare_network(args)
    snet = initialize_encoder_classifiers(args, snet)

    with evaluating(snet), torch.no_grad():
        encoder_trn_doc_embedd = args.trn_doc_embeddings
        print(f"Loaded Train Doc Embeddings")


    train_loader = prepare_data(args, encoder_trn_doc_embedd)
    clf_optimizer, combiner_optimizer, clf_scheduler, combiner_scheduler = prepare_optimizer_and_schedular(
        args, snet, len(train_loader)
    )
    train(
        args=args,
        snet=snet,
        criterion_combiner=criterion_combiner,
        criterion_clf=criterion_clf,
        clf_optimizer=clf_optimizer,
        clf_scheduler=clf_scheduler,
        combiner_optimizer=combiner_optimizer,
        combiner_scheduler=combiner_scheduler,
        train_loader=train_loader,
        trn_doc_embedds=encoder_trn_doc_embedd,
    )

    if args.save_model:
        snet.eval()
        state_dict = {}
        for k, v in snet.state_dict().items():
            state_dict[k.replace("module.", "")] = v
        torch.save(state_dict, f"{args.model_dir}/state_dict.pt")
        with open(f"{args.model_dir}/executed_script.py", "w") as fout:
            print(inspect.getsource(sys.modules[__name__]), file=fout)
        with open(f"{args.model_dir}/executed_script_args.txt", "w") as fout:
            print(args, file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wandb-project-name", type=str, help="Project Name in wandb")
    parser.add_argument("-work-dir", type=str, help="Work dir")
    parser.add_argument(
        "-dataset", type=str, help="Dataset name", default="LF-AmazonTitles-131K"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="The number of epochs to run for", default=600
    )
    parser.add_argument(
        "--combiner-start-epoch", type=int, help="The epoch to start training combiner", default=100
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, help="The batch size", default=4096
    )
    parser.add_argument(
        "-m",
        "--margin",
        type=float,
        help="Margin below which negative labels are not penalized for combiner representation",
        default=1.0,
    )
    parser.add_argument(
        "--num-negatives", type=int, help="Number of negatives to use for combined representation",
    )
    parser.add_argument(
        "--clf-margin",
        type=float,
        help="Margin below which negative labels are not penalized for classifiers",
        default=1.0,
    )
    parser.add_argument(
        "--clf-num-negatives",
        type=int,
        help="Number of negatives to use for classifiers",
        default=5,
    )
    parser.add_argument("-A", type=float, help="The propensity factor A", default=0.55)
    parser.add_argument("-B", type=float, help="The propensity factor B", default=1.5)
    parser.add_argument("--device", type=str, help="device to run", default="cuda")
    parser.add_argument("-lr", type=float, help="learning rate", default=0.0002)
    parser.add_argument("-clf-lr", type=float, help="classifiers learning rate", default=0.001)
    parser.add_argument("--pred-mode", type=str, help="ova or anns", default="ova")
    parser.add_argument(
        "--huge",
        action="store_true",
        help="Compute only recall; don't save train; use memmap",
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Should the model be saved"
    )
    parser.add_argument(
        "--trn-lbl-fname",
        type=str,
        required=False,
        help="Train label file name",
        default="trn_X_Y.txt",
    )
    parser.add_argument(
        "--file-type", type=str, required=False, help="File type txt/npz", default="txt"
    )
    parser.add_argument("--version", type=str, help="Version of the run", default="0")
    parser.add_argument(
        "--trn-filter-labels",
        type=str,
        help="filter labels for train accuracy",
        default="",
    )
    parser.add_argument(
        "--tst-seen-filter-labels",
        type=str,
        help="filter labels at seen test accuracy",
        default=""
    )
    parser.add_argument(
        "--tst-unseen-filter-labels",
        type=str,
        help="filter labels at unseen test accuracy",
        default=""
    )
    parser.add_argument(
        "--curr-steps",
        type=str,
        help="double cluster size at each step (csv)",
        default="",
    )
    parser.add_argument(
        "--eval-epochs",
        type=str,
        help="The list of epochs to evaluate at",
        default="0,19,39,59,79,99,100,102,104,106,108,111,115,119,124,129",
    )
    parser.add_argument(
        "--num-violators",
        action="store_true",
        help="Should average number of violators be printed",
    )
    parser.add_argument("--k", type=int, help="k for recall", default=100)
    parser.add_argument(
        "--max-length", type=int, help="Max length for tokenizer", default=32
    )
    parser.add_argument(
        "--transform-dim",
        type=int,
        help="Transform bert embeddings to size",
        default=-1,
    )
    parser.add_argument("--num-clusters", type=int, help="num_clusters for clf warmup epochs", default=8192) 
    parser.add_argument("--clf-cl-size", type=int, help="cluster size during Classifier Training", default=8)
    parser.add_argument("--combiner-cl-size", type=int, help="Cluster Size during Joint Training", default=8)
    parser.add_argument("--combiner-num-clusters", type=int, help="num_clusters for joint training", default=8192)
    parser.add_argument("--clf-cl-start", type=int, help="", default=999999)
    parser.add_argument("--combiner-cl-start", type=int, help="", default=999999)
    parser.add_argument("--cl-update", type=int, help="", default=5)

    # ZSXC specific parameters
    parser.add_argument(
        "--num-lbls", type=int, help="Number of Seen Labels in Train Data"
    )
    parser.add_argument(
        "--num-encoder-layers", type=int, help="Number of Encoder Layers in Combiner"
    )
    parser.add_argument(
        "--num-heads", type=int, help="Number of Heads in Self-Attention Combiner Layer"
    )
    parser.add_argument(
        "--dropout", type=float, help="The Dropout value in Combiner Encoder Layers", default=0.1
    )
    parser.add_argument(
        "--neighbor-itself",
        action="store_true",
        help="Whether to include the label itself as its first neighbor or not.",
    )
    parser.add_argument(
        "--cls-devices",
        type=str,
        help="The device to use for clustering",
        default="",
    )
    parser.add_argument(
        "--num-neighbors", type=int, help="Number Of Neighbors to Consider in Combiner"
    )
    parser.add_argument(
        "--seen-neighbors-filepath",
        type=str,
        help="Full Filepath of the threshold neighbors file (only seen)",
        default=""
    )
    parser.add_argument(
        "--seen-neighbors-scores-filepath",
        type=str,
        help="Full Filepath of the threshold neighbors scores file (only seen)",
    )
    parser.add_argument(
        "--unseen-encoder-neighbors-filepath",
        type=str,
        help="Neighbors of Unseen Labels based on Encoder Embeddings",
    )
    parser.add_argument(
        "--unseen-lbl-one-doc-embeddings-filepath",
        type=str,
        help="Encoder Document Embeddings of Documents for each Unseen Label",
    )
    parser.add_argument(
        "--trn-doc-embeddings-filepath",
        type=str,
        help="Filepath of Train Document Embeddings",
    )
    parser.add_argument(
        "--seen-tst-doc-embeddings-filepath",
        type=str,
        help="Filepath of Seen Test Document Embeddings",
    )
    parser.add_argument(
        "--unseen-tst-doc-embeddings-filepath",
        type=str,
        help="Filepath of Unseen Test Document Embeddings",
    )
    parser.add_argument(
        "--unnorm-seen-lbl-embeddings-filepath",
        type=str,
        help="Filepath of Unnormalized Seen Label Embeddings",
    )
    parser.add_argument(
        "--unnorm-unseen-lbl-embeddings-filepath",
        type=str,
        help="Filepath of Unnormalized Unseen Label Embeddings",
    )
    parser.add_argument(
        "--init-classifiers",
        action="store_true",
        help="Whether or not to initialize the classifiers with label embeddings",
    )
    parser.add_argument(
        "--num-unseen-lbls",
        type=int,
        help="Number of Unseen Labels for Unseen Evaluation in between training",
    )
    parser.add_argument(
        "--init-classifiers-path",
        type=str,
        help="Path of classifier weights to load from",
        default="",
    )
    parser.add_argument(
        "--freeze-classifiers",
        action="store_true",
        help="Whether the classifiers are frozen or not. However the Encoder transform layer is kept trainable in both cases.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="The dimension of the embeddings",
        default=768,
    )
    parser.add_argument(
        "--base-encoder",
        type=str,
        help="The base encoder to log in wandb",
        default="NGAME",
    )
    args = parser.parse_args()
    print(args)
    main(args)
