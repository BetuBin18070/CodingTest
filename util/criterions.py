import torch.nn as nn

class base_mse_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,student_pred, teacher_pred, mask, args):
        target = teacher_pred
        pred = student_pred


        if args.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


