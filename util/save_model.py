from pathlib import Path
import torch

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


def save_model(args, epoch, model, optimizer):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            'args': args,
        }
        save_on_master(to_save, checkpoint_path)


def load_model(args,optimizer):
        checkpoint = torch.load(args.resume, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])


