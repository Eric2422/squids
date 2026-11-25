from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy


def delta(
        theta_1: npt.NDArray,
        theta_2: npt.NDArray,
        magnetic_potential: Callable[[npt.NDArray], npt.NDArray],
        length: Callable[[npt.NDArray], npt.NDArray]
) -> npt.NDArray:
    return (
        theta_1 - theta_2 -
        2 * scipy.constants.e / scipy.constants.hbar *
        scipy.integrate.quad(
            np.dot(magnetic_potential(np.zeros(3)), length(np.zeros(3)))
        )
    )


def current(delta: npt.NDArray, critical_current: npt.NDArray) -> npt.NDArray:
    return critical_current * np.sin(delta)


def delta_deriv(potential: npt.NDArray) -> npt.NDArray:
    return 2 * scipy.constants.e / scipy.constants.hbar * potential


def josephson_inductance():
    pass
