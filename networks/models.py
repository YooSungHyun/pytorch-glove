from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class GloVeModel(nn.Module):
    """Implement GloVe model with Pytorch"""

    def __init__(self, embedding_size, vocab_size, x_max=100, alpha=3 / 4):
        super(GloVeModel, self).__init__()
        # 임베딩 된 중심 단어와 주변 단어 벡터의 내적이 전체 코퍼스에서의 동시 등장 확률이 되도록 만드는 것
        self.alpha = alpha
        self.x_max = x_max

        self._focal_embeddings = nn.Embedding(vocab_size, embedding_size)
        self._context_embeddings = nn.Embedding(vocab_size, embedding_size)
        self._focal_biases = nn.Embedding(vocab_size, 1)
        self._context_biases = nn.Embedding(vocab_size, 1)

    def _loss(self, focal_embed, context_embed, focal_bias, context_bias, prob):
        # count weight factor
        weight_factor = torch.pow(prob / self.x_max, self.alpha)
        weight_factor[weight_factor > 1] = 1

        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_cooccurrences = torch.log(prob)

        distance_expr = (embedding_products + focal_bias.squeeze() + context_bias.squeeze() - log_cooccurrences) ** 2

        single_losses = weight_factor * distance_expr
        # 배치의 평균
        mean_loss = torch.mean(single_losses)

        return mean_loss

    def forward(self, center, k):
        focal_embed = self._focal_embeddings(center)
        context_embed = self._context_embeddings(k)
        focal_bias = self._focal_biases(center)
        context_bias = self._context_biases(k)
        return focal_embed, context_embed, focal_bias, context_bias
