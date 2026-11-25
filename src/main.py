from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy


M_FLUX_QUANTUM = scipy.constants.value("mag. flux quantum")


def delta(
        theta_1: npt.NDArray,
        theta_2: npt.NDArray,
        magnetic_potential: Callable[[npt.NDArray], npt.NDArray],
        length: Callable[[npt.NDArray], npt.NDArray]
) -> npt.NDArray:
    """_summary_

    Parameters
    ----------
    theta_1 : npt.NDArray
        _description_
    theta_2 : npt.NDArray
        _description_
    magnetic_potential : Callable[[npt.NDArray], npt.NDArray]
        _description_
    length : Callable[[npt.NDArray], npt.NDArray]
        _description_

    Returns
    -------
    npt.NDArray
        _description_
    """
    return (
        theta_1 - theta_2 -
        2 * scipy.constants.e / scipy.constants.hbar *
        scipy.integrate.quad(
            np.dot(magnetic_potential(np.zeros(3)), length(np.zeros(3))),
            1,
            2
        )
    )


def current(delta: npt.NDArray, critical_current: np.float64) -> npt.NDArray:
    return critical_current * np.sin(delta)


def delta_deriv(potential: npt.NDArray) -> npt.NDArray:
    return 2 * scipy.constants.e / scipy.constants.hbar * potential


def josephson_inductance(critical_current: np.float64, delta: npt.NDArray):
    return M_FLUX_QUANTUM / (2 * scipy.constants.pi * critical_current * np.cos(delta))
