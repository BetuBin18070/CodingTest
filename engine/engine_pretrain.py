import math
import sys
from typing import Iterable
from tqdm import tqdm
import torch



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None)->float:
    model.train(True)

    epoch_loss = 0
    trian_loop = tqdm(data_loader)
    for samples, targets in trian_loop:

        samples = samples.to(device, non_blocking=True)
        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value

    print("Loss is {}".format(epoch_loss / len(data_loader)))
    return epoch_loss / len(data_loader)


