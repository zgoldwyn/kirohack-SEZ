"""Dataset loading and shard partitioning for the Worker.

Supports the core MVP datasets (MNIST, Fashion-MNIST, synthetic) and
provides deterministic shard partitioning so that each Worker in a
distributed job trains on a disjoint, reproducible slice of the data.

CIFAR-10 is a stretch goal and not included in the MVP.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset, TensorDataset

import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = frozenset({"MNIST", "Fashion-MNIST", "synthetic"})


def load_dataset(
    dataset_name: str,
    shard_index: int,
    shard_count: int,
    *,
    data_dir: str = "./data",
) -> Dataset:
    """Load a dataset and return the shard assigned to this Worker.

    Parameters
    ----------
    dataset_name:
        One of ``MNIST``, ``Fashion-MNIST``, or ``synthetic``.
    shard_index:
        Zero-based index of the shard assigned to this Worker.
    shard_count:
        Total number of shards the dataset is split into.  Must be > 0.
    data_dir:
        Local directory used to cache downloaded datasets.

    Returns
    -------
    torch.utils.data.Dataset
        The shard partition as a PyTorch ``Dataset``.

    Raises
    ------
    ValueError
        If *dataset_name* is not supported, or if *shard_index* / *shard_count*
        are out of range.
    """
    _validate_shard_params(shard_index, shard_count)

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name!r}. "
            f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    if dataset_name == "MNIST":
        full_dataset = _load_mnist(data_dir)
    elif dataset_name == "Fashion-MNIST":
        full_dataset = _load_fashion_mnist(data_dir)
    elif dataset_name == "synthetic":
        full_dataset = _generate_synthetic()
    else:
        # Defensive — should never reach here after the check above.
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    return _partition_dataset(full_dataset, shard_index, shard_count)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

_TORCHVISION_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def _load_mnist(data_dir: str) -> Dataset:
    """Download (if needed) and return the full MNIST training set."""
    return torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=_TORCHVISION_TRANSFORM,
    )


def _load_fashion_mnist(data_dir: str) -> Dataset:
    """Download (if needed) and return the full Fashion-MNIST training set."""
    return torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=_TORCHVISION_TRANSFORM,
    )


def _generate_synthetic(
    num_samples: int = 10_000,
    input_size: int = 784,
    num_classes: int = 10,
    seed: int = 42,
) -> Dataset:
    """Generate a deterministic synthetic classification dataset.

    Produces random input tensors and integer class labels suitable for
    training an MLP.  The fixed *seed* ensures reproducibility across
    Workers so that shard partitioning is consistent.

    Parameters
    ----------
    num_samples:
        Total number of samples to generate.
    input_size:
        Dimensionality of each input vector (default matches flattened 28×28).
    num_classes:
        Number of distinct class labels.
    seed:
        Random seed for reproducibility.
    """
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(num_samples, input_size, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    return TensorDataset(inputs, labels)


# ---------------------------------------------------------------------------
# Shard partitioning
# ---------------------------------------------------------------------------


def _partition_dataset(
    dataset: Dataset,
    shard_index: int,
    shard_count: int,
) -> Dataset:
    """Deterministically partition *dataset* and return the requested shard.

    The dataset is split into *shard_count* contiguous, non-overlapping
    slices.  If the dataset size is not evenly divisible, earlier shards
    receive one extra sample (standard balanced partitioning).

    The partitioning is purely index-based and does not shuffle, so it is
    deterministic regardless of platform or random state.
    """
    total = len(dataset)  # type: ignore[arg-type]
    indices = _shard_indices(total, shard_index, shard_count)
    return Subset(dataset, indices)


def _shard_indices(total: int, shard_index: int, shard_count: int) -> list[int]:
    """Return the list of dataset indices belonging to the given shard.

    Uses a balanced partitioning scheme: if ``total`` is not evenly
    divisible by ``shard_count``, the first ``total % shard_count`` shards
    each receive one extra sample.

    Examples
    --------
    >>> _shard_indices(10, 0, 3)
    [0, 1, 2, 3]
    >>> _shard_indices(10, 1, 3)
    [4, 5, 6]
    >>> _shard_indices(10, 2, 3)
    [7, 8, 9]
    """
    base_size = total // shard_count
    remainder = total % shard_count

    # Shards 0..remainder-1 get (base_size + 1) samples each.
    if shard_index < remainder:
        start = shard_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (shard_index - remainder) * base_size
        end = start + base_size

    return list(range(start, end))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_shard_params(shard_index: int, shard_count: int) -> None:
    """Raise ``ValueError`` if shard parameters are invalid."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be > 0, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count - 1}], got {shard_index}"
        )
