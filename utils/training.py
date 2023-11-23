from collections import defaultdict
from tqdm import tqdm
import time

import numpy as np
import torch
from torch.optim import Adam, lr_scheduler

from utils import so3, torus


def get_optimizer(*, model, lr, weight_decay, scheduler, scheduler_mode, scheduler_patience):
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    if scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7,
                                                   patience=scheduler_patience, min_lr=lr / 100)
    else:
        scheduler = None

    return optimizer, scheduler


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


def loss_tr_rot_tor(tr_pred, rot_pred, tor_pred, data, t_to_sigma,
                    tr_weight=1, rot_weight=1, tor_weight=1, apply_mean=True):
    complex_t = data.complex_t.cpu()
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(complex_t, complex_t, complex_t)
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = data.tr_score.cpu()
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * tr_sigma ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = data.rot_score.cpu()
    rot_score_norm = so3.score_norm(rot_sigma).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    edge_tor_sigma = torch.from_numpy(np.concatenate(data.tor_sigma_edge))
    tor_score = data.tor_score.cpu()
    tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.numpy())).float()
    tor_loss = ((tor_pred.cpu() - tor_score) ** 2 / tor_score_norm2)
    tor_base_loss = (tor_score ** 2 / tor_score_norm2).detach()
    if apply_mean:
        tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), \
            tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
    else:
        index = data['ligand'].batch[data['ligand', 'ligand'].edge_index[0][data['rotation_edge_mask']]].cpu()
        num_graphs = data.num_graphs
        t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
        c.index_add_(0, index, torch.ones(tor_loss.shape))
        c = c + 0.0001
        t_l.index_add_(0, index, tor_loss)
        t_b_l.index_add_(0, index, tor_base_loss)
        tor_loss, tor_base_loss = t_l / c, t_b_l / c

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight
    metrics = {
        "loss": loss.detach().numpy(),
        "tr_loss": tr_loss.detach().numpy(), "rot_loss": rot_loss.detach().numpy(), "tor_loss": tor_loss.detach().numpy(),
        "tr_base_loss": tr_base_loss.numpy(), "rot_base_loss": rot_base_loss.numpy(), "tor_base_loss": tor_base_loss.numpy()
    }
    return loss, metrics


class Meter:
    def __init__(self):
        self.dict_ = defaultdict(list)

    def add(self, **kwargs):
        for id_, values in kwargs.items():
            values = values.flatten()
            if values.size:
                self.dict_[id_].append(values)

    def summary(self):
        dict_avg, dict_q10, dict_q50, dict_q90 = dict(), dict(), dict(), dict()
        for id_, values in self.dict_.items():
            values = np.concatenate(values, axis=0)
            dict_avg[id_] = np.nanmean(values)
            dict_q10[id_] = np.nanpercentile(values, 10)
            dict_q50[id_] = np.nanpercentile(values, 50)
            dict_q90[id_] = np.nanpercentile(values, 90)
        return {"mean": dict_avg, "q10": dict_q10, "q50": dict_q50, "q90": dict_q90}


def train_epoch(model, loader, loss_fn, optimizer, t_to_sigma, ema_weights=None, logger=None, with_tqdm=False):
    model.train()
    meter = Meter()
    if with_tqdm:
        loader = tqdm(loader)
    for data in loader:
        if data.num_graphs == 1:
            continue
        optimizer.zero_grad()
        try:
            tr_pred, rot_pred, tor_pred = model(data)
            loss, metrics = loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma)
            meter.add(**metrics)
            loss.backward()
            optimizer.step()
            if ema_weights is not None:
                ema_weights.update(model.parameters())
        except Exception as e:
            if logger is not None:
                logger.warning(f'error during the training epoch ({e}), skipping batch')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            time.sleep(1)
            continue

    return meter.summary()


def val_epoch(model, loader, loss_fn, t_to_sigma, logger=None, with_tqdm=False):
    model.eval()
    meter = Meter()
    if with_tqdm:
        loader = tqdm(loader)
    for data in loader:
        try:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred = model(data)
            loss, metrics = loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma)
            meter.add(**metrics)

        except Exception as e:
            if logger is not None:
                logger.warning(f'error during the training epoch ({e}), skipping batch')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            time.sleep(1)
            continue

    return meter.summary()
