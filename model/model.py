import torch.nn as nn
# The model architecture is inspired and adapted from the MLP used in https://github.com/fdbtrs/Self-restrained-Triplet-Loss for masked face recognition

class MLP(nn.Module):
    def __init__(self, embedding_size=512):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, embedding):
        return self.model(embedding)