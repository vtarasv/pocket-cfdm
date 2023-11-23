import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import BatchNorm as e3nnBatchNorm

from params import DEVICE
from utils import SinusoidalEmbedding, so3, torus


# noinspection PyArgumentList
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, cat_dims, cont_feats, sigma_embed_dim):
        super(AtomEncoder, self).__init__()
        self.embedding_list = torch.nn.ModuleList()
        self.num_cat_feats = len(cat_dims)
        self.num_cont_feats = cont_feats
        self.sigma_embed_dim = sigma_embed_dim

        assert self.num_cat_feats > 0
        for dim in cat_dims:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

        if self.num_cont_feats:
            self.linear_cont = torch.nn.Linear(self.num_cont_feats + emb_dim, emb_dim)

        self.linear_sigma = torch.nn.Linear(self.sigma_embed_dim, emb_dim)

    def forward(self, *, x_cat=None, x_cont=None, sigma_emb=None):
        x = 0
        assert x_cat.shape[1] == self.num_cat_feats
        assert sigma_emb.shape[1] == self.sigma_embed_dim

        for i in range(self.num_cat_feats):
            x += self.embedding_list[i](x_cat[:, i])
        x += self.linear_sigma(sigma_emb)
        if self.num_cont_feats:
            assert x_cont.shape[1] == self.num_cont_feats
            x = torch.cat([x, x_cont], axis=1)
            x = self.linear_cont(x)
        return x


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.tp.weight_numel)
        )
        self.batch_norm = e3nnBatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm is not None:
            out = self.batch_norm(out)
        return out


class FitModel(torch.nn.Module):
    def __init__(self, *, t_to_sigma, ns, nv, sh_lmax, dropout, num_conv_layers, tp_batch_norm,
                 sigma_embed_dim, sigma_embed_scale, distance_embed_dim, cross_distance_embed_dim,
                 lig_cat_dims, lig_cont_feats, lig_max_radius, lig_edge_features,
                 prot_cat_dims, prot_cont_feats,
                 cross_max_radius, center_max_radius, scale_by_sigma):
        super(FitModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.ns = ns
        self.lig_max_radius = lig_max_radius
        self.lig_edge_features = lig_edge_features
        self.scale_by_sigma = scale_by_sigma

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        self.timestep_emb_func = SinusoidalEmbedding(embedding_dim=sigma_embed_dim, embedding_scale=sigma_embed_scale)

        self.lig_node_embedding = \
            AtomEncoder(emb_dim=ns, cat_dims=lig_cat_dims, cont_feats=lig_cont_feats, sigma_embed_dim=sigma_embed_dim)
        lig_edge_dim = lig_edge_features + sigma_embed_dim + distance_embed_dim
        self.lig_edge_embedding = \
            nn.Sequential(nn.Linear(lig_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)

        self.prot_node_embedding = \
            AtomEncoder(emb_dim=ns, cat_dims=prot_cat_dims, cont_feats=prot_cont_feats, sigma_embed_dim=sigma_embed_dim)
        atom_edge_dim = sigma_embed_dim + distance_embed_dim
        self.prot_edge_embedding = \
            nn.Sequential(nn.Linear(atom_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))

        cross_edge_dim = sigma_embed_dim + cross_distance_embed_dim
        self.cross_edge_embedding = \
            nn.Sequential(nn.Linear(cross_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_radius, cross_distance_embed_dim)

        irrep_seq = [f'{ns}x0e',
                     f'{ns}x0e + {nv}x1o',
                     f'{ns}x0e + {nv}x1o + {nv}x1e',
                     f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o']
        lig_conv_layers, prot_conv_layers, lig_to_prot_conv_layers, prot_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {'in_irreps': in_irreps, 'sh_irreps': self.sh_irreps, 'out_irreps': out_irreps,
                          'n_edge_features': 3 * ns, 'hidden_features': 3 * ns,
                          'residual': False, 'batch_norm': tp_batch_norm, 'dropout': dropout}
            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            prot_layer = TensorProductConvLayer(**parameters)
            prot_conv_layers.append(prot_layer)
            lig_to_prot_layer = TensorProductConvLayer(**parameters)
            lig_to_prot_conv_layers.append(lig_to_prot_layer)
            prot_to_lig_layer = TensorProductConvLayer(**parameters)
            prot_to_lig_conv_layers.append(prot_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.prot_conv_layers = nn.ModuleList(prot_conv_layers)
        self.lig_to_prot_conv_layers = nn.ModuleList(lig_to_prot_conv_layers)
        self.prot_to_lig_conv_layers = nn.ModuleList(prot_to_lig_conv_layers)

        # translation + rotation
        center_edge_dim = distance_embed_dim + sigma_embed_dim
        self.center_edge_embedding = \
            nn.Sequential(nn.Linear(center_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_radius, distance_embed_dim)
        parameters = {"in_irreps": self.lig_conv_layers[-1].out_irreps, "sh_irreps": self.sh_irreps,
                      "out_irreps": f'2x1o + 2x1e', "n_edge_features": 2 * ns,
                      "residual": False, "dropout": dropout, "batch_norm": tp_batch_norm}
        self.final_conv = TensorProductConvLayer(**parameters)
        self.tr_final_layer = \
            nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rot_final_layer = \
            nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

        # torsion
        self.final_edge_embedding = \
            nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        parameters = {"in_irreps": self.lig_conv_layers[-1].out_irreps, "sh_irreps": self.final_tp_tor.irreps_out,
                      "out_irreps": f'{ns}x0o + {ns}x0e', "n_edge_features": 3 * ns,
                      "residual": False, "dropout": dropout, "batch_norm": tp_batch_norm}
        self.tor_bond_conv = TensorProductConvLayer(**parameters)
        self.tor_final_layer = \
            nn.Sequential(nn.Linear(2 * ns, ns, bias=False), nn.Tanh(), nn.Dropout(dropout), nn.Linear(ns, 1, bias=False))

    def forward(self, data):
        data.to(DEVICE)
        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(data.complex_t, data.complex_t, data.complex_t)

        lig_node_attr_cat, lig_node_attr_sigma_emb, lig_edge_index, lig_edge_attr, lig_edge_sh = \
            self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(x_cat=lig_node_attr_cat, sigma_emb=lig_node_attr_sigma_emb)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        prot_node_attr_cat, prot_node_attr_sigma_emb, prot_edge_index, prot_edge_attr, prot_edge_sh = \
            self.build_prot_conv_graph(data)
        prot_src, prot_dst = prot_edge_index
        prot_node_attr = self.prot_node_embedding(x_cat=prot_node_attr_cat, sigma_emb=prot_node_attr_sigma_emb)
        prot_edge_attr = self.prot_edge_embedding(prot_edge_attr)

        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_prot = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for l_ in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = \
                torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = \
                self.lig_conv_layers[l_](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # inter graph message passing
            prot_to_lig_edge_attr_ = \
                torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], prot_node_attr[cross_prot, :self.ns]], -1)
            lig_inter_update = \
                self.prot_to_lig_conv_layers[l_](prot_node_attr, cross_edge_index, prot_to_lig_edge_attr_,
                                                 cross_edge_sh, out_nodes=lig_node_attr.shape[0])

            last_layer = (l_ == len(self.lig_conv_layers) - 1)
            prot_intra_update, prot_inter_update = None, None
            if not last_layer:
                prot_edge_attr_ = \
                    torch.cat([prot_edge_attr, prot_node_attr[prot_src, :self.ns], prot_node_attr[prot_dst, :self.ns]], -1)
                prot_intra_update = \
                    self.prot_conv_layers[l_](prot_node_attr, prot_edge_index, prot_edge_attr_, prot_edge_sh)

                lig_to_prot_edge_attr_ = \
                    torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], prot_node_attr[cross_prot, :self.ns]], -1)
                prot_inter_update = \
                    self.lig_to_prot_conv_layers[l_](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_prot_edge_attr_,
                                                     cross_edge_sh, out_nodes=prot_node_attr.shape[0])

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if not last_layer:
                prot_node_attr = F.pad(prot_node_attr, (0, prot_intra_update.shape[-1] - prot_node_attr.shape[-1]))
                prot_node_attr = prot_node_attr + prot_intra_update + prot_inter_update

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t)

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(DEVICE)

        if data['rotation_edge_mask'].sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=DEVICE)

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_src, tor_dst = tor_bonds
        tor_bond_vec = data['ligand'].pos[tor_dst] - data['ligand'].pos[tor_src]
        tor_bond_attr = lig_node_attr[tor_src] + lig_node_attr[tor_dst]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                      out_nodes=data['rotation_edge_mask'].sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['rotation_edge_mask']]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float().to(DEVICE))

        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t)

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch).long()
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1)
        edge_attr = torch.cat([data['ligand', 'ligand'].edge_attr,
                               torch.zeros(radius_edges.shape[-1], self.lig_edge_features, device=DEVICE)], 0)

        # compute initial features
        src, dst = edge_index
        edge_sigma_emb = data['ligand'].node_sigma_emb[src]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr_cat = data['ligand'].x_cat
        # node_attr_cont = data['ligand'].x_cont
        node_attr_sigma_emb = data['ligand'].node_sigma_emb

        edge_vec = data['ligand'].pos[dst] - data['ligand'].pos[src]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr_cat, node_attr_sigma_emb, edge_index, edge_attr, edge_sh

    def build_prot_conv_graph(self, data):
        # build the graph between receptor atoms
        data['protein'].node_sigma_emb = self.timestep_emb_func(data['protein'].node_t)
        node_attr_cat = data['protein'].x_cat
        node_attr_sigma_emb = data['protein'].node_sigma_emb

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['protein', 'protein'].edge_index
        src, dst = edge_index
        edge_vec = data['protein'].pos[dst] - data['protein'].pos[src]

        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['protein'].node_sigma_emb[src]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr_cat, node_attr_sigma_emb, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['protein'].pos / cross_distance_cutoff[data['protein'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['protein'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['protein'].pos, data['ligand'].pos, cross_distance_cutoff,
                                data['protein'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['protein'].pos[dst] - data['ligand'].pos[src]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0),
                                torch.arange(len(data['ligand'].batch)).to(DEVICE).unsqueeze(0)], dim=0)

        center_pos = torch.zeros((data.num_graphs, 3)).to(DEVICE)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst] - center_pos[src]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[dst]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['rotation_edge_mask']].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius,
                            batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh


class FitModelAlt(torch.nn.Module):
    def __init__(self, *, t_to_sigma, ns, nv, sh_lmax, dropout, num_conv_layers, tp_batch_norm,
                 sigma_embed_dim, sigma_embed_scale, distance_embed_dim, cross_distance_embed_dim,
                 lig_cat_dims, lig_cont_feats, lig_max_radius, lig_edge_features,
                 prot_cat_dims, prot_cont_feats,
                 cross_max_radius, center_max_radius, scale_by_sigma):
        super(FitModelAlt, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.ns = ns
        self.lig_max_radius = lig_max_radius
        self.lig_edge_features = lig_edge_features
        self.scale_by_sigma = scale_by_sigma

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        self.timestep_emb_func = SinusoidalEmbedding(embedding_dim=sigma_embed_dim, embedding_scale=sigma_embed_scale)

        self.lig_node_embedding = \
            AtomEncoder(emb_dim=ns, cat_dims=lig_cat_dims, cont_feats=lig_cont_feats, sigma_embed_dim=sigma_embed_dim)
        lig_edge_dim = lig_edge_features + sigma_embed_dim + distance_embed_dim
        self.lig_edge_embedding = \
            nn.Sequential(nn.Linear(lig_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)

        self.prot_node_embedding = \
            AtomEncoder(emb_dim=ns, cat_dims=prot_cat_dims, cont_feats=prot_cont_feats, sigma_embed_dim=sigma_embed_dim)
        atom_edge_dim = sigma_embed_dim + distance_embed_dim
        self.prot_edge_embedding = \
            nn.Sequential(nn.Linear(atom_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))

        cross_edge_dim = sigma_embed_dim + cross_distance_embed_dim
        self.cross_edge_embedding = \
            nn.Sequential(nn.Linear(cross_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_radius, cross_distance_embed_dim)

        irrep_seq = [f'{ns}x0e',
                     f'{ns}x0e + {nv}x1o',
                     f'{ns}x0e + {nv}x1o + {nv}x1e',
                     f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o']
        lig_conv_layers, prot_conv_layers, lig_to_prot_conv_layers, prot_to_lig_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {'in_irreps': in_irreps, 'sh_irreps': self.sh_irreps, 'out_irreps': out_irreps,
                          'n_edge_features': 3 * ns, 'hidden_features': 3 * ns,
                          'residual': False, 'batch_norm': tp_batch_norm, 'dropout': dropout}
            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            prot_layer = TensorProductConvLayer(**parameters)
            prot_conv_layers.append(prot_layer)
            lig_to_prot_layer = TensorProductConvLayer(**parameters)
            lig_to_prot_conv_layers.append(lig_to_prot_layer)
            prot_to_lig_layer = TensorProductConvLayer(**parameters)
            prot_to_lig_conv_layers.append(prot_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.prot_conv_layers = nn.ModuleList(prot_conv_layers)
        self.lig_to_prot_conv_layers = nn.ModuleList(lig_to_prot_conv_layers)
        self.prot_to_lig_conv_layers = nn.ModuleList(prot_to_lig_conv_layers)

        # translation + rotation
        center_edge_dim = distance_embed_dim + sigma_embed_dim
        self.center_edge_embedding = \
            nn.Sequential(nn.Linear(center_edge_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_radius, distance_embed_dim)
        parameters = {"in_irreps": self.lig_conv_layers[-1].out_irreps, "sh_irreps": self.sh_irreps,
                      "out_irreps": f'2x1o + 2x1e', "n_edge_features": 2 * ns,
                      "residual": False, "dropout": dropout, "batch_norm": tp_batch_norm}
        self.final_conv = TensorProductConvLayer(**parameters)
        self.tr_final_layer = \
            nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rot_final_layer = \
            nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

        # torsion
        self.final_edge_embedding = \
            nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
        parameters = {"in_irreps": self.lig_conv_layers[-1].out_irreps, "sh_irreps": self.final_tp_tor.irreps_out,
                      "out_irreps": f'{ns}x0o + {ns}x0e', "n_edge_features": 3 * ns,
                      "residual": False, "dropout": dropout, "batch_norm": tp_batch_norm}
        self.tor_bond_conv = TensorProductConvLayer(**parameters)
        self.tor_final_layer = \
            nn.Sequential(nn.Linear(2 * ns, ns, bias=False), nn.Tanh(), nn.Dropout(dropout), nn.Linear(ns, 1, bias=False))

    def forward(self, data):
        data.to(DEVICE)
        # import time
        # st = time.perf_counter()

        data['ligand'].arange = torch.arange(len(data['ligand'].batch)).to(DEVICE)

        data.center_pos = torch.zeros((data.num_graphs, 3)).to(DEVICE)
        data.center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        data.center_pos = data.center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1).to(DEVICE)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(data.complex_t, data.complex_t, data.complex_t)

        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['rotation_edge_mask']]
        tor_scale = torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float().to(DEVICE))
        rot_scale = so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(DEVICE)
        tr_scale = tr_sigma.unsqueeze(1)

        empty_ = torch.empty(0, device=DEVICE)

        no_rot = (data['rotation_edge_mask'].sum() == 0).item()

        # en = time.perf_counter()
        # print("S0", en - st)

        lig_node_attr_cat, lig_node_attr_sigma_emb, lig_edge_index, lig_edge_attr, lig_edge_sh = \
            self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(x_cat=lig_node_attr_cat, sigma_emb=lig_node_attr_sigma_emb)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        prot_node_attr_cat, prot_node_attr_sigma_emb, prot_edge_index, prot_edge_attr, prot_edge_sh = \
            self.build_prot_conv_graph(data)
        prot_src, prot_dst = prot_edge_index
        prot_node_attr = self.prot_node_embedding(x_cat=prot_node_attr_cat, sigma_emb=prot_node_attr_sigma_emb)
        prot_edge_attr = self.prot_edge_embedding(prot_edge_attr)

        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_prot = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        # en = time.perf_counter()
        # print("S1", en-st)

        for l_ in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = \
                torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = \
                self.lig_conv_layers[l_](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # inter graph message passing
            prot_to_lig_edge_attr_ = \
                torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], prot_node_attr[cross_prot, :self.ns]], -1)
            lig_inter_update = \
                self.prot_to_lig_conv_layers[l_](prot_node_attr, cross_edge_index, prot_to_lig_edge_attr_,
                                                 cross_edge_sh, out_nodes=lig_node_attr.shape[0])

            last_layer = (l_ == len(self.lig_conv_layers) - 1)
            prot_intra_update, prot_inter_update = None, None
            if not last_layer:
                prot_edge_attr_ = \
                    torch.cat([prot_edge_attr, prot_node_attr[prot_src, :self.ns], prot_node_attr[prot_dst, :self.ns]], -1)
                prot_intra_update = \
                    self.prot_conv_layers[l_](prot_node_attr, prot_edge_index, prot_edge_attr_, prot_edge_sh)

                lig_to_prot_edge_attr_ = \
                    torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], prot_node_attr[cross_prot, :self.ns]], -1)
                prot_inter_update = \
                    self.lig_to_prot_conv_layers[l_](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_prot_edge_attr_,
                                                     cross_edge_sh, out_nodes=prot_node_attr.shape[0])

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if not last_layer:
                prot_node_attr = F.pad(prot_node_attr, (0, prot_intra_update.shape[-1] - prot_node_attr.shape[-1]))
                prot_node_attr = prot_node_attr + prot_intra_update + prot_inter_update

        # en = time.perf_counter()
        # print("S2", en - st)

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        # en = time.perf_counter()
        # print("S3.1", en - st)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        # en = time.perf_counter()
        # print("S3.2", en - st)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        # en = time.perf_counter()
        # print("S3.3", en - st)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
        # en = time.perf_counter()
        # print("S3.4", en - st)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t)
        # en = time.perf_counter()
        # print("S3.5", en - st)

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))
        # en = time.perf_counter()
        # print("S3.6", en - st)

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_scale
            rot_pred = rot_pred * rot_scale
        # en = time.perf_counter()
        # print("S3.6.1", en - st)

        if no_rot:
            return tr_pred, rot_pred, empty_

        # en = time.perf_counter()
        # print("S3.6.2", en - st)

        # en = time.perf_counter()
        # print("S3.7", en - st)
        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        # en = time.perf_counter()
        # print("S3.7.1", en - st)
        tor_src, tor_dst = tor_bonds
        tor_bond_vec = data['ligand'].pos[tor_dst] - data['ligand'].pos[tor_src]
        # en = time.perf_counter()
        # print("S3.7.2", en - st)
        tor_bond_attr = lig_node_attr[tor_src] + lig_node_attr[tor_dst]
        # en = time.perf_counter()
        # print("S3.8", en - st)

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                      out_nodes=data['rotation_edge_mask'].sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        # en = time.perf_counter()
        # print("S3.9", en - st)

        if self.scale_by_sigma:
            tor_pred = tor_pred * tor_scale

        # en = time.perf_counter()
        # print("S3", en - st)

        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t)

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch).long()
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1)
        edge_attr = torch.cat([data['ligand', 'ligand'].edge_attr,
                               torch.zeros(radius_edges.shape[-1], self.lig_edge_features, device=DEVICE)], 0)

        # compute initial features
        src, dst = edge_index
        edge_sigma_emb = data['ligand'].node_sigma_emb[src]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr_cat = data['ligand'].x_cat
        # node_attr_cont = data['ligand'].x_cont
        node_attr_sigma_emb = data['ligand'].node_sigma_emb

        edge_vec = data['ligand'].pos[dst] - data['ligand'].pos[src]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr_cat, node_attr_sigma_emb, edge_index, edge_attr, edge_sh

    def build_prot_conv_graph(self, data):
        # build the graph between receptor atoms
        data['protein'].node_sigma_emb = self.timestep_emb_func(data['protein'].node_t)
        node_attr_cat = data['protein'].x_cat
        node_attr_sigma_emb = data['protein'].node_sigma_emb

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['protein', 'protein'].edge_index
        src, dst = edge_index
        edge_vec = data['protein'].pos[dst] - data['protein'].pos[src]

        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['protein'].node_sigma_emb[src]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr_cat, node_attr_sigma_emb, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['protein'].pos / cross_distance_cutoff[data['protein'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['protein'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['protein'].pos, data['ligand'].pos, cross_distance_cutoff,
                                data['protein'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['protein'].pos[dst] - data['ligand'].pos[src]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # import time
        # builds the filter and edges for the convolution generating translational and rotational scores
        # st = time.perf_counter()
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0),
                                data['ligand'].arange.unsqueeze(0)], dim=0)

        # en = time.perf_counter()
        # print("S3.1.1", en - st)
        center_pos = data.center_pos
        # en = time.perf_counter()
        # print("S3.1.2", en - st)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst] - center_pos[src]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[dst]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # en = time.perf_counter()
        # print("S3.1.3", en - st)

        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # import time
        # st = time.perf_counter()
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['rotation_edge_mask']].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius,
                            batch_x=data['ligand'].batch, batch_y=bond_batch)
        # en = time.perf_counter()
        # print("S3.7.0.1", en - st)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # en = time.perf_counter()
        # print("S3.7.0.2", en - st)

        return bonds, edge_index, edge_attr, edge_sh
