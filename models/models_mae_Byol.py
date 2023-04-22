from torch import nn

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, teacher_model, student_model):
    # the architecture of the student and teacher is same .
    for current_params, ma_params in zip(student_model.parameters(), teacher_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class BYOL(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            moving_average_decay=0.99,
            norm_pix_loss=False
    ):
        super().__init__()
        self.student = student.train(True)
        self.teacher = teacher.eval().requires_grad_(False)

        self.target_ema_updater = EMA(moving_average_decay)

        self.norm_pix_loss=norm_pix_loss

    def update_moving_average(self):
        update_moving_average(self.target_ema_updater, self.teacher, self.student)

    def forward_loss(self, student_pred, teacher_pred, mask):

        target = teacher_pred
        pred = student_pred

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
            self,
            imgs,
            mask_ratio
    ):
        self.teacher.is_teacher=True

        teacher_pred = self.teacher(imgs, mask_ratio=0)

        student_pred, mask , img_loss= self.student(imgs, mask_ratio=mask_ratio)

        kd_loss = self.forward_loss(student_pred, teacher_pred.detach_(), mask)

        loss=(kd_loss+img_loss)/2
        # with torch.no_grad():
        #     target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
        #     target_proj_one, _ = target_encoder(image_one)
        #     target_proj_two, _ = target_encoder(image_two)
        #     target_proj_one.detach_()
        #     target_proj_two.detach_()
        #
        # loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        # loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        #
        # loss = loss_one + loss_two
        return loss

