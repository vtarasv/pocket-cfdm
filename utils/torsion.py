from typing import Union

import networkx as nx
import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, HeteroData

"""
    Preprocessing and computation for torsional updates to conformers
"""


def get_torsion_mask(data: Union[Data, HeteroData]):
    g = to_networkx(data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = data["ligand", "ligand"].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i + 1, 1]

        g2 = g.to_undirected()
        g2.remove_edge(*edges[i])
        if not nx.is_connected(g2):
            l_ = sorted(nx.connected_components(g2), key=len)[0]
            l_ = list(l_)
            if len(l_) > 1:
                if edges[i, 0] in l_:
                    to_rotate.append([])
                    to_rotate.append(l_)
                else:
                    to_rotate.append(l_)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l_) == 0 else 1 for l_ in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(g.nodes())), dtype=bool)
    idx = 0
    for i in range(len(g.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def modify_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray:
        pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec)  # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy:
        pos = torch.from_numpy(pos.astype(np.float32))
    return pos
