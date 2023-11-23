import copy
import random
from typing import Callable, Tuple, Union
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from rdkit.Chem import KekulizeException, PandasTools
import torch
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.transforms import BaseTransform

from features import LigandFeaturizer
from utils import logger, so3, torus, get_torsion_mask, modify_torsion_angles, \
    axis_angle_to_matrix, rigid_transform_kabsch_3d

frag_bond_smart = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')


class PredSDFDataLoader:
    def __init__(self, df_path, pf, pocket_cent, device):
        self.df = PandasTools.LoadSDF(df_path, idName="ID", molColName="ROMol")
        self.df.apply(lambda x: x["ROMol"].SetProp("_Name", x["ID"]), axis=1)
        self.pf = pf
        self.pocket_cent = pocket_cent
        self.device = device

        self.n_samples = self.df.shape[0]

    def __iter__(self):
        self.idx_curr = 0
        return self

    def __next__(self):
        if self.idx_curr < self.n_samples:
            mol = self.df.iloc[self.idx_curr]['ROMol']
            self.idx_curr += 1
            graph = get_features_pred(mol, self.pf, self.pocket_cent)
            if graph is not None:
                graphs = [graph.to(self.device) for _ in range(1)]
                return Batch.from_data_list(graphs)
        else:
            raise StopIteration

    def __len__(self):
        return self.n_samples


def get_features_pred(mol, pf, pocket_cent):
    _id = mol.GetProp("_Name")
    try:
        mol = Chem.RemoveAllHs(mol)
        lf = LigandFeaturizer(mol).graph_feat
        lf["coords"] -= pocket_cent

        graph = get_heterograph(lf, pf)

        graph["id"] = _id
        graph["center"] = pocket_cent
        graph["rdmol"] = mol

        return graph

    except Exception as e:
        logger.warning(f"failed to generate features for {_id}, reason: {e}")


# noinspection PyTypeChecker,PyAbstractClass
def get_heterograph(lf, pf):
    graph = HeteroData()

    graph["ligand"].x_cat = torch.from_numpy(lf["atoms_feat"]).type(torch.int64)
    graph["ligand"].pos = torch.from_numpy(lf["coords"])
    graph["ligand", "covalent_contacts", "ligand"].edge_index = torch.from_numpy(lf["bonds_index"])
    graph["ligand", "covalent_contacts", "ligand"].edge_attr = torch.from_numpy(lf["bonds_type"])
    edge_mask, mask_rotate = get_torsion_mask(graph)
    graph["rotation_edge_mask"] = torch.tensor(edge_mask)
    graph["rotation_node_mask"] = mask_rotate

    graph["protein"].x_cat = torch.from_numpy(pf["atoms_feat"]).type(torch.int64)
    graph["protein"].pos = torch.from_numpy(pf["coords"])
    graph["protein", "knn_contact", "protein"].edge_index = torch.from_numpy(pf["knn_atom_index"])

    return graph


def get_rand_frag(mol, min_fraq=0.25, max_fraq=0.75):
    bonds = list(mol.GetSubstructMatches(frag_bond_smart))
    random.shuffle(bonds)
    for bond in bonds:
        em = Chem.EditableMol(copy.deepcopy(mol))
        em.RemoveBond(*bond)
        p = em.GetMol()
        try:
            Chem.SanitizeMol(p)
        except KekulizeException:
            continue
        mols = [x for x in Chem.GetMolFrags(p, asMols=True)]
        random.shuffle(mols)
        for mol_ in mols:
            na = Chem.RemoveAllHs(mol_).GetNumAtoms()
            fraq = na / Chem.RemoveAllHs(mol).GetNumAtoms()
            if (min_fraq < fraq < max_fraq) and na >= 2:
                return mol_


def save_sdf(mols, path):
    w = Chem.SDWriter(str(path))
    for mol in mols:
        w.write(mol)
    w.flush()


def rand_mol_rot(mol):
    rot = special_ortho_group.rvs(3)
    pos = mol.GetConformer().GetPositions()
    pos = pos @ rot
    pos -= pos.mean(axis=0)
    mol = set_mol_pose(mol, pos)
    return mol


def set_mol_pose(mol, pos):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (pos[i][0].item(), pos[i][1].item(), pos[i][2].item()))
    mol.AddConformer(conf)
    return mol


def parallelize(func, items):
    with mp.Pool(processes=round(mp.cpu_count()*0.8)) as pool:
        for _ in tqdm(pool.imap_unordered(func, items), total=len(items)):
            pass


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma: Callable[[float, float, float], Tuple[float, float, float]]):
        self.t_to_sigma = t_to_sigma

    def __call__(self, data: Data):
        t = np.random.uniform(low=0.0, high=1.0)
        return self.apply_noise(data, t)

    def apply_noise(self, data: Data, t: float):
        t_tr, t_rot, t_tor = t, t, t

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        data = set_time(data, t, 1)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3))
        rot_update = so3.sample_vec(eps=rot_sigma)
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['rotation_edge_mask'].sum())

        try:
            modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)
        except AssertionError:
            raise AssertionError(data["id"])
            
        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = np.ones(data['rotation_edge_mask'].sum()) * tor_sigma
        return data


def modify_conformer(data, tr_update, rot_update, torsion_updates):
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center

    if not torsion_updates.size:
        data['ligand'].pos = rigid_new_pos
        return data

    torsion_edge_index = data['ligand', 'ligand'].edge_index.T[data['rotation_edge_mask']]
    rotation_node_mask = data['rotation_node_mask']
    if isinstance(rotation_node_mask, list):
        rotation_node_mask = rotation_node_mask[0]
    flexible_new_pos = modify_torsion_angles(rigid_new_pos, torsion_edge_index,
                                             rotation_node_mask, torsion_updates)
    R, t = rigid_transform_kabsch_3d(flexible_new_pos.T, rigid_new_pos.T)
    aligned_flexible_pos = flexible_new_pos @ R.T + t.T
    data['ligand'].pos = aligned_flexible_pos
    return data


def set_time(data: Union[Data, HeteroData], t: float, batch: int):
    for node_type in data.node_types:
        data[node_type].node_t = t * torch.ones(data[node_type].num_nodes)
    data.complex_t = t * torch.ones(batch)
    return data


def randomize_position(data, tr_sigma_max):
    # randomize torsion angles
    torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=data['rotation_edge_mask'].sum())
    torsion_edge_index = data['ligand', 'ligand'].edge_index.T[data['rotation_edge_mask']]
    rotation_node_mask = data['rotation_node_mask']
    if isinstance(rotation_node_mask, list):
        rotation_node_mask = rotation_node_mask[0]
    data['ligand'].pos = \
        modify_torsion_angles(data['ligand'].pos, torsion_edge_index,
                              rotation_node_mask, torsion_updates)

    # randomize position
    molecule_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    random_rotation = torch.from_numpy(R.random().as_matrix()).float()
    data['ligand'].pos = (data['ligand'].pos - molecule_center) @ random_rotation.T

    # randomize translation
    tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
    data['ligand'].pos += tr_update

    return data
