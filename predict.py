import copy
import time
import os
from argparse import ArgumentParser
from pathlib import Path

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem

from params import DEVICE
from utils import logger,  ExponentialMovingAverage, TtoSigma, get_t_schedule, set_mol_pose
from features import lig_cat_dims, lig_cont_feats, prot_cat_dims, prot_cont_feats, PocketFeaturizer
from dataset import set_time, modify_conformer, randomize_position, PredSDFDataLoader
from model import FitModel
from rai_chem.protein import PDBParser, Protein
from rai_chem.score import get_fit_score


def main(args):
    with open("data/config/train.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open("data/config/pred.yml", "r") as f:
        pconfig = yaml.load(f, Loader=yaml.FullLoader)

    if args.samples is not None:
        pconfig["samples"] = args.samples
    if args.batch_size is not None:
        pconfig["batch_size"] = args.batch_size

    tmp_pdb_path = ".tmp_hs.pdb"
    os.system(f"reduce -Quiet -Trim {args.pdb} > .tmp.pdb")
    os.system(f"reduce -Quiet -NOFLIP .tmp.pdb > {tmp_pdb_path}")

    with open(tmp_pdb_path, "r") as f:
        pdb_lines = f.readlines()
    pocket = PDBParser(args.pdb, pdb_lines, remove_hs=False)
    pocket = Protein(args.pdb, pocket.atoms)
    pocket_cent = pocket.atoms["Coord"].mean(axis=0)
    pf = PocketFeaturizer(pocket, radius=config["prot_radius"], max_neighbors=config["prot_max_neighbors"]).graph_feat
    pf["coords"] -= pocket_cent

    loader = PredSDFDataLoader(args.sdf, pf, pocket_cent, device=DEVICE)

    logger.debug(f"using parameters: {config}")

    t_to_sigma = TtoSigma(tr_sigma_min=config["tr_sigma_min"], tr_sigma_max=config["tr_sigma_max"],
                          rot_sigma_min=config["rot_sigma_min"], rot_sigma_max=config["rot_sigma_max"],
                          tor_sigma_min=config["tor_sigma_min"], tor_sigma_max=config["tor_sigma_max"])

    t_schedule = get_t_schedule(inference_steps=pconfig["inference_steps"])

    model = FitModel(
        t_to_sigma=t_to_sigma, ns=config["ns"], nv=config["nv"], sh_lmax=2,
        dropout=config["dropout"], num_conv_layers=config["num_conv_layers"], tp_batch_norm=config["tp_batch_norm"],
        sigma_embed_dim=config["sigma_embed_dim"], sigma_embed_scale=config["sigma_embed_scale"],
        distance_embed_dim=config["distance_embed_dim"], cross_distance_embed_dim=config["cross_distance_embed_dim"],
        lig_cat_dims=lig_cat_dims, lig_cont_feats=lig_cont_feats,
        lig_max_radius=config["ligand_radius"], lig_edge_features=5,
        prot_cat_dims=prot_cat_dims, prot_cont_feats=prot_cont_feats,
        cross_max_radius=config["cross_radius"], center_max_radius=30, scale_by_sigma=True,
    ).to(DEVICE)
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f"FitModel instance created, device: {DEVICE}, n parameters: {numel}")

    ema_weights = ExponentialMovingAverage(model.parameters(), decay=config["ema_rate"])

    checkpoint_path = Path("workdir/.model_checkpoints/")
    last_model_name = "model.pt"
    checkpoint_path_last = checkpoint_path / last_model_name
    try:
        dict_ = torch.load(checkpoint_path_last, map_location=torch.device("cpu"))
        model.load_state_dict(dict_["model"], strict=True)
        if ema_weights is not None:
            ema_weights.load_state_dict(dict_["ema_weights"], device=DEVICE)
        epoch_start = dict_["epoch"] + 1
        best_val_loss = dict_["best_val_loss"]
        logger.info(f"model checkpoints in {str(checkpoint_path_last)} fond, " +
                    f"epoch {epoch_start-1}, best val loss: {best_val_loss}")
    except Exception as e:
        logger.error(f"failed to apply model checkpoints, {e}")
        raise Exception(e)

    pred(model, pocket, loader, steps=pconfig["actual_steps"], samples=pconfig["samples"],
         batch_size=pconfig["batch_size"], t_schedule=t_schedule, t_to_sigma=t_to_sigma,
         max_trials=2, save_path=args.save_path, with_tqdm=True, no_filter=args.no_filter)


# noinspection PyTypeChecker,PyArgumentList
def pred(model, prot, pred_loader, steps, samples, *, batch_size, t_schedule, t_to_sigma,
         max_trials=2, save_path, with_tqdm=False, no_filter):

    model.eval()

    sdf_writer = Chem.SDWriter(str(save_path))

    if with_tqdm:
        pred_loader = tqdm(pred_loader)
    for data in pred_loader:
        if data is None:
            continue
        _id = data["id"][0]
        if not isinstance(_id, str):
            _id = str(_id.item())
        rdmol = data["rdmol"][0]
        center = data["center"][0]
        protein_pose = data["protein"].pos.cpu().numpy() + center

        no_result, n_trials, d_time = True, 0, None
        samples_list = None
        while no_result and n_trials < max_trials:
            n_trials += 1
            try:
                st_time = time.time()
                samples_list = inference_samples(model, data, steps, samples, batch_size=batch_size,
                                                 t_schedule=t_schedule, t_to_sigma=t_to_sigma)
                en_time = time.time()
                d_time = en_time - st_time
                assert len(samples_list), "no data in output"
            except Exception as e:
                if n_trials < max_trials:
                    logger.warning(f"error during the inference ({e}), retrying complex {_id}")
                else:
                    logger.warning(f"error during the inference ({e}), skipping complex {_id}")
            else:
                no_result = False

        if no_result:
            continue

        poses, min_self_dists, min_cross_dists, poses_mol, fit_scores = [], [], [], [], []
        for sample in samples_list:
            pose = sample["ligand"].pos.cpu().numpy() + sample["center"][0]
            pose_mol = set_mol_pose(rdmol, pose)
            poses.append(pose)
            poses_mol.append(pose_mol)
            min_self_dists.append(pdist(pose, "euclidean").min())
            min_cross_dists.append(cdist(pose, protein_pose, "euclidean").min())

            pose_mol.SetProp("_Name", _id)
            fscores = get_fit_score(pose_mol, prot)
            fit_scores.append(fscores)

        df = pd.DataFrame()
        df["ROMol"] = poses_mol
        df["MolID"] = _id
        df["dTime"] = d_time

        df["MinSelfDist"] = min_self_dists
        df["MinCrossDist"] = min_cross_dists

        df = pd.concat([df, pd.DataFrame(fit_scores)], axis=1)

        df = df.sort_values("FitScore", ascending=False).reset_index(drop=True)
        df["MODEL"] = df.index + 1

        if no_filter:
            df_ = df
        else:
            df_ = df[df["FitScore"] > 0]
        for idx, row in df_.iterrows():
            mol = row["ROMol"]
            for k, v in row[1:].to_dict().items():
                mol.SetProp(k, str(v))
            sdf_writer.write(mol)

    sdf_writer.close()


# noinspection PyTypeChecker
def inference_samples(model, data, steps, samples, *, batch_size, t_schedule, t_to_sigma,
                      no_final_step_noise=True):
    samples_list = [randomize_position(copy.deepcopy(data), t_to_sigma.tr_sigma_max) for _ in range(samples)]

    for t_idx in range(steps):
        t = t_schedule[t_idx]
        tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t, t, t)
        dt = t_schedule[t_idx] - t_schedule[t_idx + 1] if t_idx < steps - 1 else t_schedule[t_idx]
        dt_tr, dt_rot, dt_tor = dt, dt, dt

        samples_loader = DataLoader(samples_list, batch_size=batch_size, shuffle=False)
        samples_list_mod = []
        for data_samples in samples_loader:
            b = data_samples.num_graphs
            data_samples = set_time(data_samples, t, b)
            try:
                with torch.no_grad():
                    tr_score, rot_score, tor_score = model(data_samples)
            except Exception as e:
                torch.cuda.empty_cache()
                raise Exception(e)
            else:
                tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(t_to_sigma.tr_sigma_max /
                                                                     t_to_sigma.tr_sigma_min)))
                rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(t_to_sigma.rot_sigma_max /
                                                                       t_to_sigma.rot_sigma_min)))
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(t_to_sigma.tor_sigma_max /
                                                                       t_to_sigma.tor_sigma_min)))

                tr_z = torch.zeros((b, 3)) if (no_final_step_noise and t_idx == steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
                rot_z = torch.zeros((b, 3)) if (no_final_step_noise and t_idx == steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()
                tor_z = torch.zeros(tor_score.shape) if (no_final_step_noise and t_idx == steps - 1) \
                    else torch.normal(mean=0, std=1, size=tor_score.shape)
                tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()

                tor_per_mol = tor_perturb.shape[0] // b

                for i, complex_graph in enumerate(data_samples.cpu().to_data_list()):
                    samples_list_mod.append(modify_conformer(complex_graph,
                                                             tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                                             tor_perturb[i * tor_per_mol:(i + 1) * tor_per_mol]))

        samples_list = samples_list_mod

    return samples_list


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--sdf", type=str, required=True)
    parser.add_argument("-s", "--save_path", type=str, required=True)

    parser.add_argument("--samples", type=int, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--no_filter", default=False, action="store_true")

    main(parser.parse_args())
