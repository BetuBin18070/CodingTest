import math
import sys
from typing import Iterable, Optional, Tuple

import torch

from timm.data import Mixup
from timm.utils import accuracy

from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    model.train(True)
    epoch_loss, epoch_acc = 0, 0

    with tqdm(data_loader, desc='Finetune Train') as trian_loop:
        for samples, targets in trian_loop:

            optimizer.zero_grad()

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)


            with torch.cuda.amp.autocast():
                outputs, _ = model(samples)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            labels = targets

            acc1 = accuracy(outputs, labels, topk=(1,))[0]

            loss_value = loss.item()
            epoch_loss += loss_value
            epoch_acc += acc1
    epoch_loss, epoch_acc = epoch_loss / len(data_loader), epoch_acc / len(data_loader)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    epoch_loss, epoch_acc = 0, 0

    with tqdm(data_loader, desc='Evaluate') as trian_loop:
        for samples, targets in trian_loop:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()
            epoch_loss += loss_value
            labels = targets

            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            epoch_acc += acc1


    epoch_loss, epoch_acc = epoch_loss / len(data_loader), epoch_acc / len(data_loader)
    return epoch_loss, epoch_acc

