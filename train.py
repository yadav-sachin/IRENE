import sys, os, time, warnings, gc

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
import wandb
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm
from xclib.utils.clustering import cluster_balance, b_kmeans_dense
from xclib.utils.matrix import SMatrix

from utils.helper_utils import load_config_and_runtime_args
from utils.cluster_gpu import balanced_cluster
from datasets import DatasetDNeighbors, collate_fn_neighbors
from nets import MetaClfGen
from utils.eval_utils import (
    timeit,
    validate,
)
from contextlib import contextmanager

args = load_config_and_runtime_args(sys.argv)


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
    """
    print("==> Creating model, optimizer...")

    net = MetaClfGen(
        num_neighbors=args.num_neighbors,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_trn_lbls=args.num_trn_lbls,
        dim=args.dim,
        dropout=args.dropout,
        device=args.device,
    )
    net.to(args.device)

    print(net)
    return net


def prepare_optimizer_and_schedular(args, net, len_train_loader):
    no_decay = ["bias", "LayerNorm.weight"]
    gp = [
        {
            "params": [
                p
                for n, p in net.combiner.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in net.combiner.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {"params": net.pos_embedd.pos_embeddings, "weight_decay": 0.0},
        {"params": net.score_embedd.score_embeddings, "weight_decay": 0.0},
    ]
    optimizer = transformers.AdamW(gp, **{"lr": args.lr, "eps": 1e-06})

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=args.num_epochs * len_train_loader,
    )
    return optimizer, scheduler


def prepare_data(args):
    print("==> Creating Dataloader...")
    print("Using Doc Side Sampling")
    train_dataset = DatasetDNeighbors(
        trn_doc_embeddings=args.trn_X_unnorm_embeddings,
        Y_trn=args.Y_trn,
        Y_trn_neighbor_indices=args.Y_trn_neighbor_indices,
        Y_trn_neighbor_scores=args.Y_trn_neighbor_scores,
        num_trn_lbls=args.num_trn_lbls,
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


def log_metrics_wandb(acc, r, prefix, epoch):
    recall_k = [1, 3, 5, 10, 30, 50, 100]
    precision_k = [1, 3, 5]
    metrics = {"epoch": epoch}
    for k in recall_k:
        metrics[f"{prefix}_Recall@{k}"] = r[k - 1]
    for k in precision_k:
        metrics[f"{prefix}_Precision@{k}"] = acc[0][k - 1]

    wandb.log(metrics)

    precision_log = f"Epoch: {epoch} {prefix}_Precision(1,3,5) : {acc[0][0]:.5f}, {acc[0][2]:.5f}, {acc[0][4]:.5f}"
    recall_log = f"Epoch: {epoch} {prefix}_Recall(1,3,5,10,30,50,100) : {r[0]:.5f}, {r[2]:.5f}, {r[4]:.5f}, {r[9]:.5f}, {r[29]:.5f}, {r[49]:.5f}, {r[99]:.5f}"

    with open(args.log_filepath, "a") as f:
        print(precision_log, file=f)
        print(recall_log, file=f)

    print(precision_log)
    print(recall_log)


def zero_shot_evaluation(args, net, epoch, device):
    """
    Evaluates the zero-shot and generalized zero-shot performance.
    Args:
    args: Argument object containing necessary parameters.
    net: The neural network model.
    epoch: Current epoch number.
    device: Device to run the model on.
    """
    Y_zero_neighbor_scores = args.Y_zero_neighbor_indices.copy()
    Y_zero_neighbor_scores[:] = 1

    acc, r = validate(
        args=args,
        net=net,
        Y_eval=args.Y_tst_zero,
        eval_neighbor_indices=args.Y_zero_neighbor_indices,
        eval_neighbor_scores=Y_zero_neighbor_scores,
        eval_doc_embeddings=args.tst_X_zero_unnorm_embeddings,
        lbl_embeddings=args.Y_zero_unnorm_embeddings,
        prefix="zero",
        mode=args.eval_mode,
        epoch=epoch,
        device=device,
    )
    log_metrics_wandb(acc=acc, r=r, prefix="zero", epoch=epoch)

    Y_full_neighbor_indices = np.concatenate(
        [args.Y_trn_neighbor_indices, args.Y_zero_neighbor_indices], axis=0
    )
    Y_full_zero_neighbor_scores = np.concatenate(
        [args.Y_trn_neighbor_scores, Y_zero_neighbor_scores], axis=0
    )

    acc, r = validate(
        args=args,
        net=net,
        Y_eval=args.Y_tst_full,
        eval_neighbor_indices=Y_full_neighbor_indices,
        eval_neighbor_scores=Y_full_zero_neighbor_scores,
        eval_doc_embeddings=args.tst_X_full_unnorm_embeddings,
        lbl_embeddings=args.Y_full_unnorm_embeddings,
        prefix="full",
        mode=args.eval_mode,
        epoch=epoch,
        device=device,
    )
    log_metrics_wandb(acc=acc, r=r, prefix="full", epoch=epoch)

def train(
    args,
    net,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    trn_doc_embeddings,
):

    zero_shot_evaluation(args=args, net=net, epoch=-1, device=args.device)
    if args.save_model:
        net.eval()
        torch.save(net.state_dict(), f"{args.OUT_DIR}/state_dict_ep_-1.pt")

    args.cl_size = 1

    for epoch in range(args.num_epochs):
        net.train()
        torch.set_grad_enabled(True)
        pbar = tqdm(train_loader)
        loss_history = []
        t1 = time.time()
        for data in pbar:
            net.zero_grad()

            ip_embeddings = data["doc_embeddings"].to(net.device)

            op_embeddings = net.forward(
                neighbors_index=data["neighbors_indices"],
                neighbors_scores=data["neighbors_scores"],
                self_and_neighbors_attention_mask=data["self_and_neighbors_mask"],
                lbl_index=data["lbl_indices"],
            )

            loss = criterion(
                ip_embeddings @ op_embeddings.T, data["Y"].float().to(args.device)
            )

            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description("epoch: {}, loss: {:4e}".format(epoch, loss.item()))

        loss_log = f"epoch {epoch} time taken {time.time() - t1:.5f} || average loss = {'{:.5e}'.format(np.mean(loss_history))}"
        with open(args.log_filepath, "a") as f:
            print(loss_log, file=f)
        print(loss_log)
        wandb.log({"loss": np.mean(loss_history), "epoch": epoch})

        if epoch == args.cl_start_ep:
            args.cl_size = args.cl_start_size
            if (
                ((epoch - args.cl_start_ep) % args.cl_update == 0)
                or (epoch == args.cl_start_ep)
            ):
                print(
                    f"Updating clusters with cluster size {args.cl_size} (using stale embeddings)"
                )
                embs = trn_doc_embeddings.copy()
                tree_depth = int(
                    np.ceil(np.log(embs.shape[0] / args.cl_size) / np.log(2))
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
            zero_shot_evaluation(args=args, net=net, epoch=epoch, device=args.device)

            if args.save_model:
                net.eval()
                torch.save(net.state_dict(), f"{args.OUT_DIR}/state_dict_ep_{epoch}.pt")


def init_classifiers_embedds(args, net):
    with evaluating(net), torch.no_grad():
        net.Y_trn_lbl_embeddings[:-1].data.copy_(
            torch.from_numpy(args.Y_trn_unnorm_embeddings)
        )
        print(f"Loaded Train Label Embeddings")

        net.Y_trn_lbl_embeddings.requires_grad = False

    with evaluating(net), torch.no_grad():
        net.Y_trn_lbl_classifiers[:-1].data.copy_(torch.from_numpy(args.Y_trn_unnorm_classifiers))
        print(f"Loaded Train Label Classifiers")

        print("Freezing Classifiers")
        net.Y_trn_lbl_classifiers.requires_grad = False

    return net


def main():
    wandb.init(
        project=f"{args.project}_{args.dataset}", name=f"{args.base_retriever}_{args.expname}", save_code=True
    )

    args.device = torch.device(args.device)
    args.DATA_DIR = f"Datasets/{args.dataset}"
    args.DATA_ASSETS_DIR = f"Dataset_Assets/{args.dataset}/{args.base_retriever}"
    args.OUT_DIR = (
        f"Results/{args.project}/{args.dataset}/{args.base_retriever}/{args.expname}"
    )

    args.log_filepath = f"{args.OUT_DIR}/log.txt"
    with open(args.log_filepath, "w") as f:
        pass

    os.makedirs(args.OUT_DIR, exist_ok=True)

    # Get GPU clustering devices
    if args.cls_devices == "":
        args.cls_devices = None
    else:
        args.cls_devices = [int(x.strip()) for x in args.cls_devices.split(",")]

    if args.eval_epochs == "":
        args.eval_epochs = None
    else:
        args.eval_epochs = set([int(x.strip()) for x in args.eval_epochs.split(",")])

    # get curriculum steps
    if args.curr_steps == "":
        args.curr_steps = set()
    else:
        args.curr_steps = set([int(x.strip()) for x in args.curr_steps.split(",")])

    args.Y_tst_zero = load_npz(os.path.join(args.DATA_DIR, args.Y_tst_zero_filename))
    args.Y_tst_full = load_npz(os.path.join(args.DATA_DIR, args.Y_tst_full_filename))
    args.Y_trn = load_npz(os.path.join(args.DATA_DIR, args.Y_trn_filename))

    criterion = prepare_loss(args, margin=args.margin, num_negatives=args.num_negatives)

    args.Y_trn_neighbor_indices = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_trn_neighbor_indices_filename}"
    )[:, : args.num_neighbors]
    args.Y_trn_neighbor_scores = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_trn_neighbor_scores_filename}"
    )[:, : args.num_neighbors]
    args.Y_zero_neighbor_indices = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_zero_neighbor_indices_filename}"
    )[:, : args.num_neighbors]

    if not args.neighbor_itself:
        args.trn_neighbors_indices = np.load(
            f"{args.DATA_ASSETS_DIR}/{args.Y_trn_neighbor_indices_filename}"
        )[:, 1 : args.num_neighbors + 1]
        args.trn_neighbors_scores = np.load(
            f"{args.DATA_ASSETS_DIR}/{args.Y_trn_neighbor_scores_filename}"
        )[:, 1 : args.num_neighbors + 1]

    args.num_trn_lbls = args.Y_trn.shape[1]

    args.Y_trn_neighbor_scores[args.Y_trn_neighbor_scores >= 10] = 10
    args.Y_trn_neighbor_indices[args.Y_trn_neighbor_scores <= 0] = args.num_trn_lbls

    args.trn_X_unnorm_embeddings = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.trn_X_unnorm_embeddings_filename}"
    ).astype(np.float32)
    args.Y_trn_unnorm_embeddings = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_trn_unnorm_embeddings_filename}"
    ).astype(np.float32)
    args.tst_X_zero_unnorm_embeddings = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.tst_X_zero_unnorm_embeddings_filename}"
    ).astype(np.float32)

    args.Y_trn_unnorm_classifiers = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_trn_unnorm_classifiers_filename}"
    ).astype(np.float32)

    args.Y_full_unnorm_embeddings = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_full_unnorm_embeddings_filename}"
    ).astype(np.float32)
    args.Y_zero_unnorm_embeddings = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.Y_zero_unnorm_embeddings_filename}"
    ).astype(np.float32)
    args.tst_X_full_unnorm_embeddings = np.load(
        f"{args.DATA_ASSETS_DIR}/{args.tst_X_full_unnorm_embeddings_filename}"
    ).astype(np.float32)

    net = prepare_network(args)
    net = init_classifiers_embedds(args, net)

    train_loader = prepare_data(args)
    optimizer, scheduler = (
        prepare_optimizer_and_schedular(args, net, len(train_loader))
    )
    train(
        args=args,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        trn_doc_embeddings=args.trn_X_unnorm_embeddings,
    )

    if args.save_model:
        net.eval()
        torch.save(net.state_dict(), f"{args.OUT_DIR}/state_dict.pt")


if __name__ == "__main__":
    main()
