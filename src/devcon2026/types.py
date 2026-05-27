"""Shared package data-structure helpers."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any
from typing import Type
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound="ArrayConvertible")


@dataclass
class ArrayConvertible:
    """Mixin to convert dataclass fields to/from numeric arrays for ODE solving."""

    def to_array(self) -> NDArray[np.float64]:
        """Serialize dataclass values to a float64 numpy vector."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def from_array(cls: Type[T], values: NDArray[np.floating[Any]]) -> T:
        """Instantiate dataclass from a positionally aligned numpy vector."""
        return cls(**{f.name: float(values[i]) for i, f in enumerate(fields(cls))})
