import jittor as jt
from jittor import nn
import numpy as np


def _normalize(x, dim=-1, eps=1e-12):
    norm = jt.sqrt((x * x).sum(dim=dim, keepdims=True) + eps)
    return x / norm


class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = jt.array((1 - (diag + l1 + l2)).astype(np.bool_))
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = jt.nn.bmm(x.unsqueeze(1), y.transpose(0, 1).unsqueeze(0).expand([x.shape[0], -1, -1]))
        v = v.squeeze(1)
        return v

    def _cosine_simililarity(self, x, y):
        x_norm = _normalize(x, dim=-1)
        y_norm = _normalize(y, dim=-1)
        v = jt.matmul(x_norm.unsqueeze(1), y_norm.unsqueeze(0).transpose(0, 2, 1))
        v = v.squeeze(1)
        return v

    def execute(self, zis, zjs):
        if self.batch_size != zis.shape[0]:
            self.batch_size = zis.shape[0]

        self.mask_samples_from_same_repr = self._get_correlated_mask().bool()
        representations = jt.concat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = jt.diag(similarity_matrix, self.batch_size)
        r_pos = jt.diag(similarity_matrix, -self.batch_size)
        positives = jt.concat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = jt.concat((positives, negatives), dim=1)
        logits = logits / self.temperature

        labels = jt.zeros(2 * self.batch_size).int64()
        loss = nn.cross_entropy_loss(logits, labels)
        loss = loss * logits.shape[0]

        return loss / (2 * self.batch_size)
