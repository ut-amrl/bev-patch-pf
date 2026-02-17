from collections.abc import Sequence
from typing import Any

from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate


def safe_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of dictionaries, skipping fields that cannot be collated"""
    output = {}
    keys = batch[0].keys()
    for key in keys:
        values = [b[key] for b in batch]
        try:
            output[key] = default_collate(values)
        except Exception:
            output[key] = values
    return output


class SubsetWithAttributes(Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)
