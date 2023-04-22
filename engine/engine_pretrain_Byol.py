import math
import sys
from typing import Iterable
from tqdm import tqdm
import torch




def train_one_epoch(model_byol: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None) -> float:

    epoch_loss = 0
    trian_loop = tqdm(data_loader)
    for samples, targets in trian_loop:
        samples = samples.to(device, non_blocking=True)
        optimizer.zero_grad()

        loss=model_byol(samples,args.mask_ratio)

        loss.backward()
        optimizer.step()

        model_byol.update_moving_average()

        loss_value = loss.item()
        epoch_loss += loss_value

    print("Loss is {}".format(epoch_loss / len(data_loader)))
    return epoch_loss / len(data_loader)
