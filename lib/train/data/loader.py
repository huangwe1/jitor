import jittor as jt
import importlib
import collections
import numpy as np
from jittor.dataset import Dataset
string_classes = str
int_classes = int
from lib.utils import TensorDict, TensorList


def ltr_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], jt.Var):
        return jt.stack(batch, 0)
    elif isinstance(batch[0], np.ndarray):
        return jt.stack([jt.array(b) for b in batch], 0)
    elif isinstance(batch[0], int_classes):
        return jt.array(np.array(batch, dtype=np.int64))
    elif isinstance(batch[0], float):
        return jt.array(np.array(batch, dtype=np.float64))
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch
    raise TypeError((error_msg.format(type(batch[0]))))


def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], jt.Var):
        return jt.stack(batch, 1)
    elif isinstance(batch[0], np.ndarray):
        return jt.stack([jt.array(b) for b in batch], 1)
    elif isinstance(batch[0], int_classes):
        return jt.array(np.array(batch, dtype=np.int64))
    elif isinstance(batch[0], float):
        return jt.array(np.array(batch, dtype=np.float64))
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        return batch
    raise TypeError((error_msg.format(type(batch[0]))))


class LTRLoader(Dataset):
    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

        self.dataset = dataset
        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim

        if collate_fn is None:
            if stack_dim == 0:
                self.collate_fn = ltr_collate
            elif stack_dim == 1:
                self.collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')
        else:
            self.collate_fn = collate_fn

        self.total_len = len(dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_batch(self, batch):
        return self.collate_fn(batch)
