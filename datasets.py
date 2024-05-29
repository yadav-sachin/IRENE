import torch
import numpy as np
import numba as nb
import random
import torch.distributed as dist
from sklearn.preprocessing import normalize
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from xclib.data import data_utils


def clip_batch_lengths(ind, mask, max_len):
    _max = min(np.max(np.sum(mask, axis=1)), max_len)
    return ind[:, :_max], mask[:, :_max]

def _collate_fn_neighbors(batch):
    batch_pos_labels = []
    random_chosen_pos_label_indices = []
    for item in batch:
        batch_pos_labels.append(item["pos_indices"])
        random_chosen_pos_label_indices.append(item["pos_ind"])

    batch_size = len(batch_pos_labels)

    neighbors_indices = np.vstack([x["neighbors_indices"] for x in batch])
    neighbors_scores = np.vstack([x["neighbors_scores"] for x in batch])
    pos_ind = np.vstack([x["pos_ind"] for x in batch])
    self_and_neighbors_mask = np.vstack(
        [x["self_and_neighbors_attn_mask"] for x in batch]
    )
    doc_embeddings = np.vstack([x["doc_embedding"] for x in batch])

    batch_selection = np.zeros((batch_size, batch_size), dtype=np.int32)

    random_chosen_pos_label_indices_set = set(random_chosen_pos_label_indices)
    random_chosen_pos_label_indices = np.array(
        random_chosen_pos_label_indices, dtype=np.int32
    )

    for (i, item) in enumerate(batch_pos_labels):
        intersection = set(item).intersection(random_chosen_pos_label_indices_set)
        result = np.zeros(batch_size, dtype=np.float32)
        for idx in intersection:
            result += idx == random_chosen_pos_label_indices
        batch_selection[i] = result

    return (
        pos_ind,
        neighbors_indices,
        neighbors_scores,
        self_and_neighbors_mask,
        doc_embeddings,
        batch_selection,
    )


def collate_fn_neighbors(batch):
    """
    collate function
    """
    batch_data = {}
    batch_size = len(batch)
    batch_data["batch_size"] = torch.tensor(batch_size, dtype=torch.int32)

    (
        pos_ind,
        neighbors_indices,
        neighbors_scores,
        self_and_neighbors_attention_mask,
        doc_embeddings,
        batch_selection,
    ) = _collate_fn_neighbors(batch)

    batch_data["indices"] = torch.LongTensor([item["index"] for item in batch])
    batch_data["lbl_indices"] = torch.from_numpy(pos_ind)
    batch_data["neighbors_indices"] = torch.from_numpy(neighbors_indices)
    batch_data["neighbors_scores"] = torch.from_numpy(neighbors_scores)
    batch_data["self_and_neighbors_mask"] = torch.from_numpy(
        self_and_neighbors_attention_mask
    ).bool()
    batch_data["doc_embeddings"] = torch.from_numpy(doc_embeddings)
    batch_data["Y"] = torch.from_numpy(batch_selection)
    batch_data["Y_mask"] = None
    return batch_data


class DatasetDNeighbors(torch.utils.data.Dataset):
    """
    Dataset for document-side training
    with Label Neighbors and Scores
    """

    def __init__(
        self,
        encoder_trn_doc_embedds,
        seen_trn_X_Y_fname,
        seen_neighbors_lbl_indices,
        seen_neighbors_scores,
        num_seen_lbls,
        num_neighbors,
    ):
        super().__init__()
        self.seen_trn_X_Y = data_utils.read_gen_sparse(seen_trn_X_Y_fname)

        self.trn_doc_embedds = normalize(encoder_trn_doc_embedds) if normalize else encoder_trn_doc_embedds
        self.seen_trn_X_Y = data_utils.read_gen_sparse(seen_trn_X_Y_fname)

        self.seen_neighbors_lbl_indices = seen_neighbors_lbl_indices
        self.seen_neighbors_scores = seen_neighbors_scores
        self.num_seen_lbls = num_seen_lbls
        self.num_neighbors = num_neighbors

        assert (
            num_seen_lbls == self.seen_trn_X_Y.shape[1]
        ), "Mismatch between the Number of seen labels and the Number of Columns of trn_X_Y (seen train)"

    def __getitem__(self, doc_index):
        """Get a Label and Neighbors at given Doc Index"""
        pos_lbl_indices = self.seen_trn_X_Y[doc_index].indices
        doc_embedding = self.trn_doc_embedds[doc_index]
        pos_chosen_lbl_indx = np.random.choice(pos_lbl_indices)

        self_and_neighbors_attn_mask = np.zeros(
            (self.num_neighbors + 1,), dtype=np.int64
        )
        self_and_neighbors_attn_mask[0] = 1
        self_and_neighbors_attn_mask[1:][
            self.seen_neighbors_lbl_indices[pos_chosen_lbl_indx] < self.num_seen_lbls
        ] = 1

        item = {
            "pos_indices": pos_lbl_indices,
            "pos_ind": pos_chosen_lbl_indx,
            "neighbors_indices": self.seen_neighbors_lbl_indices[pos_chosen_lbl_indx],
            "neighbors_scores": self.seen_neighbors_scores[pos_chosen_lbl_indx],
            "self_and_neighbors_attn_mask": self_and_neighbors_attn_mask,
            "index": doc_index,
            "doc_embedding": doc_embedding,
        }

        return item

    def __len__(self):
        return self.seen_trn_X_Y.shape[0]