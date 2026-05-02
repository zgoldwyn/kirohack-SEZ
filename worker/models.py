"""MLP model definition for the Worker.

Provides a configurable multi-layer perceptron (MLP) that can be
instantiated from the task configuration received from the Coordinator.
The architecture is parameterised by input size (derived from the dataset),
a list of hidden-layer widths, output size (number of classes), and an
activation function.

For the MVP only ``relu`` activation is supported.  The design is
intentionally simple — one ``nn.Linear`` + activation per hidden layer,
followed by a final ``nn.Linear`` projection to the output logits.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn


# ---------------------------------------------------------------------------
# Supported activations
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
}

SUPPORTED_ACTIVATIONS = frozenset(_ACTIVATIONS)


def _get_activation(name: str) -> nn.Module:
    """Return an activation module instance for the given name.

    Parameters
    ----------
    name:
        Case-insensitive activation name.  Must be one of
        :data:`SUPPORTED_ACTIVATIONS`.

    Raises
    ------
    ValueError
        If *name* is not a supported activation function.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation: {name!r}. "
            f"Supported activations: {sorted(SUPPORTED_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]()


# ---------------------------------------------------------------------------
# Dataset → input/output size helpers
# ---------------------------------------------------------------------------

# Maps dataset names to (input_size, output_size).
# Input size is the flattened feature dimension; output size is the number
# of classes.
_DATASET_SHAPES: dict[str, tuple[int, int]] = {
    "MNIST": (784, 10),
    "Fashion-MNIST": (784, 10),
    "synthetic": (784, 10),
}


def get_dataset_shape(dataset_name: str) -> tuple[int, int]:
    """Return ``(input_size, output_size)`` for a supported dataset.

    Parameters
    ----------
    dataset_name:
        One of the supported dataset names (``MNIST``, ``Fashion-MNIST``,
        ``synthetic``).

    Returns
    -------
    tuple[int, int]
        A ``(input_size, output_size)`` pair.

    Raises
    ------
    ValueError
        If *dataset_name* is not recognised.
    """
    if dataset_name not in _DATASET_SHAPES:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Known datasets: {sorted(_DATASET_SHAPES)}"
        )
    return _DATASET_SHAPES[dataset_name]


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """A configurable multi-layer perceptron.

    Parameters
    ----------
    input_size:
        Dimensionality of the input feature vector (e.g. 784 for 28×28
        images).
    hidden_layers:
        Sequence of hidden-layer widths.  For example ``[128, 64]``
        produces two hidden layers with 128 and 64 units respectively.
        An empty sequence creates a single linear layer from *input_size*
        directly to *output_size*.
    output_size:
        Number of output logits (typically the number of classes).
    activation:
        Name of the activation function applied after each hidden layer.
        Defaults to ``"relu"``.  Must be one of :data:`SUPPORTED_ACTIVATIONS`.

    Examples
    --------
    >>> model = MLP(input_size=784, hidden_layers=[128, 64], output_size=10)
    >>> import torch
    >>> x = torch.randn(4, 784)
    >>> model(x).shape
    torch.Size([4, 10])
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int] = (128, 64),
        output_size: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be > 0, got {output_size}")
        for i, h in enumerate(hidden_layers):
            if h <= 0:
                raise ValueError(
                    f"hidden_layers[{i}] must be > 0, got {h}"
                )

        layers: list[nn.Module] = []
        prev_size = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(_get_activation(activation))
            prev_size = h

        # Final projection — no activation (raw logits for CrossEntropyLoss).
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch_size, input_size)``.  If the
            input has more dimensions (e.g. ``(B, 1, 28, 28)``), it is
            automatically flattened to ``(B, -1)``.

        Returns
        -------
        torch.Tensor
            Output logits of shape ``(batch_size, output_size)``.
        """
        # Flatten spatial dims so (B, 1, 28, 28) → (B, 784).
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_model(
    dataset_name: str,
    model_type: str,
    hidden_layers: Sequence[int] = (128, 64),
    activation: str = "relu",
) -> nn.Module:
    """Build a model from high-level task configuration parameters.

    This is the primary entry point used by the training loop.  It maps
    the ``model_type`` string from the task config to the appropriate
    model class and derives ``input_size`` / ``output_size`` from the
    dataset name.

    Parameters
    ----------
    dataset_name:
        Name of the dataset (used to derive input/output dimensions).
    model_type:
        Model architecture identifier.  Currently only ``"MLP"`` is
        supported.
    hidden_layers:
        Hidden-layer widths forwarded to the model constructor.
    activation:
        Activation function name forwarded to the model constructor.

    Returns
    -------
    nn.Module
        An initialised PyTorch model ready for training.

    Raises
    ------
    ValueError
        If *model_type* is not supported or *dataset_name* is unknown.
    """
    if model_type != "MLP":
        raise ValueError(
            f"Unsupported model type: {model_type!r}. Supported: ['MLP']"
        )

    input_size, output_size = get_dataset_shape(dataset_name)

    return MLP(
        input_size=input_size,
        hidden_layers=list(hidden_layers),
        output_size=output_size,
        activation=activation,
    )
