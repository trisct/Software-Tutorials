# Tensorboard

Tensorboard is a tool for logging the training procedure of neural networks

### Installation
```
pip install tensorboard
```

### Use Tensorboard in PyTorch
```
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir = '.')
```

You can then save different types of data using the tensorboard writer.

__Scalars__
```
writer.add_scalar('scalar name', global_step=global_step)
```

An example program is in `Software-Tutorials/tensorboard/example.py`.

### Displaying
After completing logging, in bash, use
```
tensorboard --logdir .
```
to display the logged data. Tensorboard will prompt where to access them. You can download CSV format logging from the website.

### CSV Processing
A tool for proceessing the CSV data is in `Software-Tutorials/tensorboard/csv_plot.py`.