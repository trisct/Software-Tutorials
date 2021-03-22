from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir = '.')

for step in range(100):
    scalar_value = 1 / (step+1)
    print(step)
    writer.add_scalar('name', scalar_value, global_step=step)