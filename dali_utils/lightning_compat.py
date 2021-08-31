import os
import math
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy


class SimCLRWrapper(DALIGenericIterator):
    def __init__(self, transform):
        image_ids = [f"im{i}" for i in range(transform.copies)]
        if transform.stage != "inference":
            image_ids.append("label")
        super().__init__(
            transform, image_ids, last_batch_policy=LastBatchPolicy.PARTIAL
        )

        self.num_samples = transform.num_samples
        self.next_fn = self.get_next(transform.stage != "inference")

    def get_next(self, with_label):
        def include_label(self):
            out = super().__next__()
            out = out[0]
            return tuple([out[k] for k in self.output_map[:-1]]), torch.squeeze(
                out[self.output_map[-1]]
            )

        def without_label(self):
            out = super().__next__()
            out = out[0]
            return tuple([out[k] for k in self.output_map])

        if with_label:
            return include_label
        else:
            return without_label

    def __next__(self):
        return self.next_fn(self)

    def __len__(self):
        return math.ceil(self.num_samples // self.batch_size)


class ClassifierWrapper(DALIGenericIterator):
    def __init__(self, transform):
        image_ids = [f"im{i}" for i in range(transform.copies)]
        if transform.stage != "inference":
            image_ids.append("label")
        super().__init__(
            transform, image_ids, last_batch_policy=LastBatchPolicy.PARTIAL
        )

        self.num_samples = transform.num_samples
        self.next_fn = self.get_next(transform.stage != "inference")

    def get_next(self, with_label):
        def include_label(self):
            out = super().__next__()
            out = out[0]
            return out[self.output_map[0]], torch.squeeze(out[self.output_map[-1]])

        def without_label(self):
            out = super().__next__()
            out = out[0]
            return out[self.output_map[0]]

        if with_label:
            return include_label
        else:
            return without_label

    def __next__(self):
        return self.next_fn(self)

    def __len__(self):
        return math.ceil(self.num_samples // self.batch_size)
