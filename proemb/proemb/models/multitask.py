import numpy as np
import torch
import torch.nn as nn


class ProSEMT(nn.Module):
    def __init__(self, embedding, scop_predict, cmap_predict, surf_dist_predict, interface_predictor):
        super(ProSEMT, self).__init__()
        self.skipLSTM = embedding
        self.scop_predict = scop_predict
        self.cmap_predict = cmap_predict
        self.surf_dist_predict = surf_dist_predict
        self.interface_predictor = interface_predictor
    def forward(self, seq_unpacked, lens_unpacked, apply_proj=True):
        return self.skipLSTM(seq_unpacked, lens_unpacked, apply_proj)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)

    def predict_aa_distance(self, dist_info):
        return self.surf_dist_predict.predict_aa_distance(dist_info)

    def predict_aa_shape_index(self, shape_info):
        return self.surf_dist_predict.predict_aa_shape_index(shape_info)

    def predict_rsasa(self, rsasa_info):
        return self.surf_dist_predict.predict_rsasa(rsasa_info)

    def predict_aa_interface(self, surface_info):
        return self.interface_predictor.predict_aa_interface(surface_info)

    def predict_aa_distance_interface(self, dist_info):
        return self.interface_predictor.predict_aa_distance(dist_info)

    def predict_delta_sasa(self, delta_sasa_info):
        return self.interface_predictor.predict_delta_sasa(delta_sasa_info)


class SurfacePredictor(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout):
        super(SurfacePredictor, self).__init__()
        self.linear = SimpleMLP(embed_dim, hidden_dim, 1, dropout=dropout)
        self.linear_shape_index = SimpleMLP(embed_dim, hidden_dim, 1, dropout=dropout)
        self.linear_rsasa = SimpleMLP(embed_dim, hidden_dim, 1, dropout=dropout)

    def predict_aa_distance(self, dist_info):
        return self.linear(dist_info)

    def predict_aa_shape_index(self, shape_info):
        return self.linear_shape_index(shape_info)

    def predict_rsasa(self, rsasa_info):
        return self.linear_rsasa(rsasa_info)

class InterfacePredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super(InterfacePredictor, self).__init__()
        self.linear = SimpleMLP(embed_dim, hidden_dim, 1, dropout=dropout)
        self.linear_distance = SimpleMLP(embed_dim, hidden_dim, 1, dropout=dropout)
        self.linear_delta_sasa = SimpleMLP(embed_dim, hidden_dim, 1, dropout=dropout)

    def predict_aa_interface(self, interface_info):
        return self.linear(interface_info)

    def predict_aa_distance(self, dist_info):
        return self.linear_distance(dist_info)

    def predict_delta_sasa(self, delta_sasa_info):
        return self.linear_delta_sasa(delta_sasa_info)

class BilinearContactMap(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W z_j + b)
    """
    def __init__(self, embed_dim, hidden_dim=1000):
        super(BilinearContactMap, self).__init__()

        self.scale = np.sqrt(hidden_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        z_flat = z.view(-1, z.size(2))
        h = self.linear(z_flat).view(z.size(0), z.size(1), -1)
        # h changes dim to dim, and we do Z_transpose so we get s.shape = (seqL, seqL)
        s = torch.bmm(h, z.transpose(1, 2)) / self.scale + self.bias
        return s


class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1) - y), -1)


class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1) - y) ** 2, -1)


class OrdinalRegression(nn.Module):
    def __init__(self, embed_dim, n_classes, compare=L1(), align_method='ssa', beta_init=None):
        super(OrdinalRegression, self).__init__()

        self.n_in = embed_dim
        self.n_out = n_classes

        self.compare = compare
        self.align_method = align_method

        if beta_init is None:
            # set beta to expectation of comparison
            # assuming embeddings are unit normal

            if type(compare) is L1:
                ex = 2 * np.sqrt(2 / np.pi) * embed_dim  # expectation for L1
                var = 4 * (1 - 2 / np.pi) * embed_dim  # variance for L1
            elif type(compare) is L2:
                ex = 4 * embed_dim  # expectation for L2
                var = 32 * embed_dim  # variance for L2
            else:
                ex = 0
                var = embed_dim

            beta_init = ex / np.sqrt(var)

        self.theta = nn.Parameter(torch.ones(1, n_classes - 1) / np.sqrt(var))
        self.beta = nn.Parameter(torch.zeros(n_classes - 1) + beta_init)

        self.clip()

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.theta.data.clamp_(min=0)

    def forward(self, z_x, z_y):
        return self.score(z_x, z_y)

    def score(self, z_x, z_y):

        s = self.compare(z_x, z_y)

        if self.align_method == 'ssa':
            a = torch.softmax(s, 1)
            b = torch.softmax(s, 0)
            a = a + b - a * b
            a = a / torch.sum(a)
        else:
            raise Exception('Unknown alignment method: ' + self.align_method)

        a = a.view(-1, 1)
        s = s.view(-1, 1)

        # c is a single score
        c = torch.sum(a * s)
        logits = c * self.theta + self.beta
        return logits.view(-1)


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        return self.main(x)
