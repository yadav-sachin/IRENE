import torch
import torch.nn.functional as F


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


class MetaClfGen(torch.nn.Module):
    """
    A network class to support Siamese style training
    * specialized for sentence-bert or hugging face
    * hard-coded to use a joint encoder

    """

    def __init__(
        self,
        num_neighbors,
        num_layers,
        num_heads,
        num_trn_lbls,
        dim,
        dropout,
        device,
    ):
        super(MetaClfGen, self).__init__()
        self.device = device
        self.n_lbls = num_trn_lbls
        self.n_neighbors = num_neighbors
        self.dropout = dropout

        self.dim = dim

        self.Y_trn_lbl_classifiers = torch.nn.Parameter(torch.Tensor(num_trn_lbls + 1, self.dim))
        self.Y_trn_lbl_embeddings = torch.nn.Parameter(
            torch.Tensor(num_trn_lbls + 1, self.dim)
        )

        self.doc_drop_transform = torch.nn.Dropout(p=0.5)

        torch.nn.init.xavier_uniform_(self.Y_trn_lbl_classifiers)
        self.pos_embedd = PositionalEmbedding(d_model=self.dim)
        self.score_embedd = ScoreEmbedding(d_model=self.dim)

        with torch.no_grad():
            self.Y_trn_lbl_classifiers[self.n_lbls] = 0
            self.Y_trn_lbl_classifiers[self.n_lbls].requires_grad = False
            self.Y_trn_lbl_embeddings[self.n_lbls] = 0
            self.Y_trn_lbl_embeddings[self.n_lbls].requires_grad = False

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.combiner = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def label_classifier_repr(
        self,
    ):
        return self.Y_trn_lbl_classifiers[:-1]

    def encode_label_combined_repr(
        self,
        lbl_embeddings,
        neighbors_index,
        neighbors_scores,
        self_and_neighbors_attention_mask,
        lbl_index=None,
    ):
        neighbors_scores = neighbors_scores.to(self.device)
        neighbors_index, self_and_neighbors_attention_mask = neighbors_index.to(
            self.device
        ), self_and_neighbors_attention_mask.to(self.device)

        if lbl_index is not None:
            lbl_index = lbl_index.to(self.device)
            lbl_embedd = F.embedding(lbl_index.squeeze(1), self.Y_trn_lbl_embeddings)
        else:
            lbl_embedd = lbl_embeddings.to(self.device)

        neighbors_clfs = F.embedding(neighbors_index, self.Y_trn_lbl_classifiers)

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
                combined_reprs
                * self_and_neighbors_attention_mask.unsqueeze(2)[
                    :, : combined_reprs.shape[1]
                ],
                dim=1,
            )
            / torch.sum(self_and_neighbors_attention_mask, dim=1)[:, None]
        )
        return combined_reprs

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
        return self.dim
