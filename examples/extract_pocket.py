from rdkit import Chem

from rai_chem.protein import PDBParser, Protein
from rai_chem.ligand import Ligand
from rai_chem.utils import get_close_coords


def filt_prot(protein_path, ligand_path, pocket_path, radius=6):
    with open(protein_path, "r") as f:
        pdb_lines = f.readlines()
    protein = PDBParser(protein_path, pdb_lines, remove_hs=False)
    prot = Protein(protein_path, protein.atoms)

    rdligand = next(Chem.SDMolSupplier(ligand_path, removeHs=False))
    lig = Ligand(ligand_path, rdligand)

    lc, pc = get_close_coords(lig.atoms["Coord"], prot.atoms["Coord"], radius)
    lines_filt = []
    res_keep = set(prot.atoms[pc]["ResidueID"].tolist())
    for l_str in pdb_lines:
        if l_str.startswith("ATOM  "):
            resid = l_str[21].strip() + l_str[22:26].strip() + l_str[26].strip() + l_str[17:20].strip()
            if resid in res_keep:
                lines_filt.append(l_str)

    with open(pocket_path, "w") as f:
        f.writelines(lines_filt)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--protein_path", type=str, required=True, help="Path to input protein file (PDB)")
    parser.add_argument("--ligand_path", type=str, required=True, help="Path to input known ligand file (SDF)")
    parser.add_argument("--pocket_path", type=str, required=True, help="Path to save binding pocket file (PDB)")
    parser.add_argument("--radius", type=float, required=False, default=6)
    args = parser.parse_args()

    filt_prot(args.protein_path, args.ligand_path, args.pocket_path, radius=args.radius)
