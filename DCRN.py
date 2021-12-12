import opt
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter


# AE encoder from DFCN
class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


# AE decoder from DFCN
class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


# Auto Encoder from DFCN
class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)


# GNNLayer from DFCN
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if opt.args.name == "dblp":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        elif opt.args.name == "cite" or opt.args.name == "acm":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if opt.args.name == "dblp":
                support = self.act(F.linear(features, self.weight))
            elif opt.args.name == "cite" or opt.args.name == "acm":
                support = self.act(torch.mm(features, self.weight))
        else:
            if opt.args.name == "dblp":
                support = F.linear(features, self.weight)
            elif opt.args.name == "cite" or opt.args.name == "acm":
                support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az


# IGAE encoder from DFCN
class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1, az_1 = self.gnn_1(x, adj, active=True)
        z_2, az_2 = self.gnn_2(z_1, adj, active=True)
        z_igae, az_3 = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z_1, z_2, z_igae]


# IGAE decoder from DFCN
class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1, az_1 = self.gnn_4(z_igae, adj, active=True)
        z_2, az_2 = self.gnn_5(z_1, adj, active=True)
        z_hat, az_3 = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z_1, z_2, z_hat]


# Improved Graph Auto Encoder from DFCN
class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE, self).__init__()
        # IGAE encoder
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input)

        # IGAE decoder
        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)


# readout function
class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        # cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))
        Z_tilde = torch.cat(Z_tilde, dim=0)
        return Z_tilde.view(1, -1)


# Dual Correlation Reduction Network
class DCRN(nn.Module):
    def __init__(self, n_node=None):
        super(DCRN, self).__init__()

        # Auto Encoder
        self.ae = AE(
            ae_n_enc_1=opt.args.ae_n_enc_1,
            ae_n_enc_2=opt.args.ae_n_enc_2,
            ae_n_enc_3=opt.args.ae_n_enc_3,
            ae_n_dec_1=opt.args.ae_n_dec_1,
            ae_n_dec_2=opt.args.ae_n_dec_2,
            ae_n_dec_3=opt.args.ae_n_dec_3,
            n_input=opt.args.n_input,
            n_z=opt.args.n_z)

        # Improved Graph Auto Encoder From DFCN
        self.gae = IGAE(
            gae_n_enc_1=opt.args.gae_n_enc_1,
            gae_n_enc_2=opt.args.gae_n_enc_2,
            gae_n_enc_3=opt.args.gae_n_enc_3,
            gae_n_dec_1=opt.args.gae_n_dec_1,
            gae_n_dec_2=opt.args.gae_n_dec_2,
            gae_n_dec_3=opt.args.gae_n_dec_3,
            n_input=opt.args.n_input)

        # fusion parameter from DFCN
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))

        # cluster layer (clustering assignment matrix)
        self.cluster_centers = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)

        # readout function
        self.R = Readout(K=opt.args.n_clusters)

    # calculate the soft assignment distribution Q
    def q_distribute(self, Z, Z_ae, Z_igae):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by IGAE
        Returns:
            the soft assignment distribution Q
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, X_tilde1, Am, X_tilde2, Ad):
        # node embedding encoded by AE
        Z_ae1 = self.ae.encoder(X_tilde1)
        Z_ae2 = self.ae.encoder(X_tilde2)

        # node embedding encoded by IGAE
        Z_igae1, A_igae1, AZ_1, Z_1 = self.gae.encoder(X_tilde1, Am)
        Z_igae2, A_igae2, AZ_2, Z_2 = self.gae.encoder(X_tilde2, Ad)

        # cluster-level embedding calculated by readout function
        Z_tilde_ae1 = self.R(Z_ae1)
        Z_tilde_ae2 = self.R(Z_ae2)
        Z_tilde_igae1 = self.R(Z_igae1)
        Z_tilde_igae2 = self.R(Z_igae2)

        # linear combination of view 1 and view 2
        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2

        # node embedding fusion from DFCN
        Z_i = self.a * Z_ae + self.b * Z_igae
        Z_l = torch.spmm(Am, Z_i)
        S = torch.mm(Z_l, Z_l.t())
        S = F.softmax(S, dim=1)
        Z_g = torch.mm(S, Z_l)
        Z = self.alpha * Z_g + Z_l

        # AE decoding
        X_hat = self.ae.decoder(Z)

        # IGAE decoding
        Z_hat, Z_adj_hat, AZ_de, Z_de = self.gae.decoder(Z, Am)
        sim = (A_igae1 + A_igae2) / 2
        A_hat = sim + Z_adj_hat

        # node embedding and cluster-level embedding
        Z_ae_all = [Z_ae1, Z_ae2, Z_tilde_ae1, Z_tilde_ae2]
        Z_gae_all = [Z_igae1, Z_igae2, Z_tilde_igae1, Z_tilde_igae2]

        # the soft assignment distribution Q
        Q = self.q_distribute(Z, Z_ae, Z_igae)

        # propagated embedding AZ_all and embedding Z_all
        AZ_en = []
        Z_en = []
        for i in range(len(AZ_1)):
            AZ_en.append((AZ_1[i]+AZ_2[i])/2)
            Z_en.append((Z_1[i]+Z_2[i])/2)
        AZ_all = [AZ_en, AZ_de]
        Z_all = [Z_en, Z_de]

        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all
