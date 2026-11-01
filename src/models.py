"""PyTorch dataset, padding, and BiLSTM-CRF components."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchcrf import CRF


class SequenceTaggingDataset(Dataset):
    def __init__(self, data, word2idx, pos2idx, tag2idx):
        self.data = data
        self.word2idx = word2idx
        self.pos2idx = pos2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        words = [self.word2idx.get(word.lower(), self.word2idx["<UNK>"]) for word in item["words"]]
        pos = [self.pos2idx.get(tag, self.pos2idx["<UNK>"]) for tag in item["pos_tags"]]
        tags = [self.tag2idx[tag] for tag in item["bio_tags"]]
        return (
            torch.tensor(words, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(tags, dtype=torch.long),
        )


def pad_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_length = max(len(words) for words, _, _ in batch)
    batch_size = len(batch)
    words_padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    pos_padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    tags_padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    for row, (words, pos, tags) in enumerate(batch):
        length = len(words)
        words_padded[row, :length] = words
        pos_padded[row, :length] = pos
        tags_padded[row, :length] = tags
        mask[row, :length] = True
    return words_padded, pos_padded, tags_padded, mask


class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        pos_size,
        num_tags,
        word_embedding_dim=128,
        pos_embedding_dim=32,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for a bidirectional LSTM.")
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_size, pos_embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            word_embedding_dim + pos_embedding_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def emissions(self, words, pos):
        embedded = torch.cat((self.word_embedding(words), self.pos_embedding(pos)), dim=-1)
        encoded, _ = self.lstm(self.dropout(embedded))
        return self.classifier(self.dropout(encoded))

    def forward_loss(self, words, pos, tags, mask):
        emissions = self.emissions(words, pos)
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, words, pos, mask):
        return self.crf.decode(self.emissions(words, pos), mask=mask)

