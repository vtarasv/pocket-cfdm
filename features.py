import numpy as np
import torch
from torch_cluster import radius_graph
from rdkit.Chem.rdchem import Mol, BondType

from rai_chem.protein import Protein


lig_feats_allow = {
    "Symbol": ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "misc"],
    "TotalDegree": [0, 1, 2, 3, 4, "misc"],
    "TotalValence": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "TotalNumHs": [0, 1, 2, 3, 4, "misc"],
    "FormalCharge": [-1, 0, 1, "misc"],
    "Hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "IsAromatic": [False, True],
    "NumRings": [0, 1, 2, 3, "misc"],
    "IsInRing3": [False, True],
    "IsInRing4": [False, True],
    "IsInRing5": [False, True],
    "IsInRing6": [False, True],
    "IsInRing7": [False, True],
    "IsInRing8": [False, True],
}

lig_cat_dims = list(map(len, [
    lig_feats_allow["Symbol"],
    lig_feats_allow["TotalDegree"],
    lig_feats_allow["TotalValence"],
    lig_feats_allow["TotalNumHs"],
    lig_feats_allow["FormalCharge"],
    lig_feats_allow["Hybridization"],
    lig_feats_allow["IsAromatic"],
    lig_feats_allow["NumRings"],
    lig_feats_allow["IsInRing3"],
    lig_feats_allow["IsInRing4"],
    lig_feats_allow["IsInRing5"],
    lig_feats_allow["IsInRing6"],
    lig_feats_allow["IsInRing7"],
    lig_feats_allow["IsInRing8"],

]))
lig_cont_feats = 0

lig_bonds = {BondType.UNSPECIFIED: 0, BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}

prot_feats_allow = {
    "Symbol": ["C", "N", "O", "S"],
    "TotalDegree": [0, 1, 2, 3, 4],
    "TotalNumHs": [0, 1, 2, 3, 4],
    "Hybridization": ["SP2", "SP3"],
    "IsHydrophobe": [0, 1],
    "IsHDonor": [0, 1],
    "IsWeakHDonor": [0, 1],
    "IsHAcceptor": [0, 1],
    "IsPositive": [0, 1],
    "IsNegative": [0, 1],
    "InAromatic": [0, 1],
    "InAmide": [0, 1],
    "AtomName": ["C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2",
                 "CH2", "CZ", "CZ2", "CZ3", "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
                 "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG"],
    "ResidueName": ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"],
}

prot_cat_dims = list(map(len, [
    prot_feats_allow["Symbol"],
    prot_feats_allow["TotalDegree"],
    prot_feats_allow["TotalNumHs"],
    prot_feats_allow["Hybridization"],
    prot_feats_allow["IsHydrophobe"],
    prot_feats_allow["IsHDonor"],
    prot_feats_allow["IsWeakHDonor"],
    prot_feats_allow["IsHAcceptor"],
    prot_feats_allow["IsPositive"],
    prot_feats_allow["IsNegative"],
    prot_feats_allow["InAromatic"],
    prot_feats_allow["InAmide"],
    prot_feats_allow["AtomName"],
    prot_feats_allow["ResidueName"],
]))
prot_cont_feats = 0


def safe_index(l_, e):
    try:
        return l_.index(e)
    except ValueError:
        return len(l_) - 1


class LigandFeaturizer:
    def __init__(self, mol: Mol):
        self.mol = mol
        self.graph_feat = dict()
        self.get_features()

    def get_features(self):
        self.graph_feat["coords"] = self.mol.GetConformer().GetPositions().astype(np.float32)
        self.graph_feat["atoms_feat"] = self.get_atom_features(self.mol)
        self.graph_feat["bonds_index"], self.graph_feat["bonds_type"] = self.get_edges(self.mol)

    @staticmethod
    def get_atom_features(mol):
        ringinfo = mol.GetRingInfo()
        atom_features_list = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_features_list.append([
                safe_index(lig_feats_allow["Symbol"], atom.GetSymbol()),
                safe_index(lig_feats_allow["TotalDegree"], atom.GetTotalDegree()),
                safe_index(lig_feats_allow["TotalValence"], atom.GetTotalValence()),
                safe_index(lig_feats_allow["TotalNumHs"], atom.GetTotalNumHs()),
                safe_index(lig_feats_allow["FormalCharge"], atom.GetFormalCharge()),
                safe_index(lig_feats_allow["Hybridization"], str(atom.GetHybridization())),
                lig_feats_allow["IsAromatic"].index(atom.GetIsAromatic()),
                safe_index(lig_feats_allow["NumRings"], ringinfo.NumAtomRings(idx)),
                lig_feats_allow["IsInRing3"].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
                lig_feats_allow["IsInRing4"].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
                lig_feats_allow["IsInRing5"].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
                lig_feats_allow["IsInRing6"].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
                lig_feats_allow["IsInRing7"].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
                lig_feats_allow["IsInRing8"].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            ])

        return np.array(atom_features_list, dtype=np.float32)

    @staticmethod
    def get_edges(mol):
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [lig_bonds[bond.GetBondType()]]

        edge_index = np.array([row, col], dtype=np.int64)
        edge_type = np.array(edge_type, dtype=np.int64)
        edge_attr = np.zeros((edge_type.size, len(lig_bonds)), dtype=np.float32)
        edge_attr[np.arange(edge_type.size), edge_type] = 1

        return edge_index, edge_attr


class PocketFeaturizer:
    def __init__(self, prot: Protein, radius: float = None, max_neighbors: int = None):
        self.prot = prot
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.graph_feat = dict()
        self.get_features()

    def get_features(self):
        self.graph_feat["coords"] = self.prot.atoms["Coord"].copy()
        self.graph_feat["atoms_feat"] = self.get_atom_features(self.prot)
        self.graph_feat["knn_atom_index"] = radius_graph(torch.from_numpy(self.prot.atoms["Coord"]), self.radius,
                                                         max_num_neighbors=self.max_neighbors).numpy().astype(np.int64)

    @staticmethod
    def get_atom_features(prot):
        atom_features_list = []
        for atom in prot.atoms:
            atom_features_list.append([
                prot_feats_allow["Symbol"].index(atom["Symbol"]),
                prot_feats_allow["TotalDegree"].index(atom["TotalDegree"]),
                prot_feats_allow["TotalNumHs"].index(atom["TotalNumHs"]),
                prot_feats_allow["Hybridization"].index(atom["Hybridization"]),
                prot_feats_allow["IsHydrophobe"].index(atom["IsHydrophobe"]),
                prot_feats_allow["IsHDonor"].index(atom["IsHDonor"]),
                prot_feats_allow["IsWeakHDonor"].index(atom["IsWeakHDonor"]),
                prot_feats_allow["IsHAcceptor"].index(atom["IsHAcceptor"]),
                prot_feats_allow["IsPositive"].index(atom["IsPositive"]),
                prot_feats_allow["IsNegative"].index(atom["IsNegative"]),
                prot_feats_allow["InAromatic"].index(atom["InAromatic"]),
                prot_feats_allow["InAmide"].index(atom["InAmide"]),
                prot_feats_allow["AtomName"].index(atom["AtomName"]),
                prot_feats_allow["ResidueName"].index(atom["ResidueName"]),
            ])

        return np.array(atom_features_list, dtype=np.float32)
