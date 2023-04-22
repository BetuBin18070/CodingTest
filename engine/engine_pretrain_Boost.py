import math
import sys
from typing import Iterable
from tqdm import tqdm
import torch



def train_one_epoch(model_student: torch.nn.Module, model_teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, criterion,
                    log_writer=None,
                    args=None) -> float:
    model_student.train(True)
    model_teacher.eval().requires_grad_(False)

    epoch_loss = 0
    trian_loop = tqdm(data_loader)
    for samples, targets in trian_loop:
        samples = samples.to(device, non_blocking=True)
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        teacher_pred = model_teacher(samples, mask_ratio=0)

        student_pred, mask ,loss_img = model_student(samples, mask_ratio=args.mask_ratio)

        loss_kd = criterion(student_pred, teacher_pred.detach_(), mask, args)

        loss=(loss_kd+loss_img)/2

        loss.backward()
        optimizer.step()


        loss_value = loss.item()
        epoch_loss += loss_value

    print("Loss is {}".format(epoch_loss / len(data_loader)))
    return epoch_loss / len(data_loader)
