"""Shared data model used by the algorithm modules and the GUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Param:
    """Description of one GUI parameter of an operation."""

    key: str
    label: str
    kind: str = "int"          # int | float | choice | bool
    default: Any = 0
    minimum: float = 0
    maximum: float = 100
    step: float = 1
    decimals: int = 2
    choices: tuple = ()
    suffix: str = ""


@dataclass
class OpResult:
    """What an operation returns to the GUI."""

    image: np.ndarray
    log: str = ""
    kernel: np.ndarray | None = None
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Operation:
    """One algorithm in the catalogue."""

    key: str
    name: str
    category: str
    description: str
    func: Callable
    params: tuple = ()
    pointwise: bool = False
