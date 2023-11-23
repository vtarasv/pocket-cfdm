import copy
import os
import pickle
from pathlib import Path
import logging
import datetime
import multiprocessing as mp
import signal
from contextlib import contextmanager

import numpy as np
import rdkit
from rdkit import Chem
from spyrmsd import rmsd, molecule

log_path = Path(__file__).parents[1] / "workdir" / "log"
os.makedirs(log_path, exist_ok=True)
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d_%H-%M-%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(os.path.join(log_path, f"LOG_{now}.txt"))
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

c_format = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
                             '%H:%M:%S')
f_format = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
                             '%Y-%m-%d %H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_strings_from_txt(path):
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def parallelize(func, data):
    result = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for processed_batch in pool.imap_unordered(func, chunker(data, size=4)):
            result.append(processed_batch)

    return result


def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]


def chunker_df(df, size):
    for pos in range(0, df.shape[0], size):
        yield df.iloc[pos:pos + size]


def set_mol_pose(mol: rdkit.Chem.rdchem.Mol, pos: np.ndarray):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (pos[i][0].item(), pos[i][1].item(), pos[i][2].item()))
    mol.AddConformer(conf)
    return mol


def get_symmetry_rmsd(mol, coords1, coords2):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        rmsd_ = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol.atomicnums,
            mol.adjacency_matrix,
            mol.adjacency_matrix,
        )
        return rmsd_


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
