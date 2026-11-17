import numpy as np
import numpy.typing as npt
import scipy


def delta(theta_1: npt.NDArray, theta_2: npt.NDArray, magnetic_potential: npt.NDArray) -> npt.NDArray:
    return theta_1 - theta_2 - 2 * scipy.constants.e / scipy.constants.hbar


def delta_deriv(potential: npt.NDArray) -> npt.NDArray:
    return 2 * scipy.constants.e / scipy.constants.hbar * potential
