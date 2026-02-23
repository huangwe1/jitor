import os
from collections import OrderedDict

try:
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from torch.utils.tensorboard import SummaryWriter
except:
    try:
        from tensorboardX import SummaryWriter
    except:
        class SummaryWriter:
            """Fallback no-op TensorboardWriter when tensorboard is unavailable."""
            def __init__(self, log_dir=None, **kwargs):
                self.log_dir = log_dir
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
            def add_scalar(self, tag, scalar_value, global_step=None):
                pass
            def add_text(self, tag, text_string, global_step=None):
                pass
            def close(self):
                pass


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        self.directory = directory
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, script_name, description):
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)
