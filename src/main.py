from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy


M_FLUX_QUANTUM = scipy.constants.value("mag. flux quantum")


def delta(
        theta_1: npt.NDArray,
        theta_2: npt.NDArray,
        magnetic_potential: Callable[[npt.NDArray, np.float64], npt.NDArray],
        length: Callable[[npt.NDArray], npt.NDArray]
) -> npt.NDArray:
    """Return the gauge-invariant phase difference between the two superconductors.

    Parameters
    ----------
    theta_1 : :class:`npt.NDArray`
        The phase of the first superconductor.
    theta_2 : :class:`npt.NDArray`
        The phase of the second superconductor.
    magnetic_potential : Callable[[:class:`npt.NDArray`, :class:`np.float64`], :class:`npt.NDArray`]
        The magnetic potential passing through the SQUID.
    length : Callable[[:class:`npt.NDArray`], :class:`npt.NDArray`]
        The length to integrate the magnetic potential along.

    Returns
    -------
    :class:`npt.NDArray`
        The gauge-invariant phase difference between the two superconductors.
    """
    return (
        theta_1 - theta_2 -
        2 * scipy.constants.e / scipy.constants.hbar *
        scipy.integrate.quad(
            np.dot(magnetic_potential(np.zeros(3),
                   np.float64(0)), length(np.zeros(3))),
            1,
            2
        )
    )


def current(delta: npt.NDArray, critical_current: np.float64) -> npt.NDArray:
    """The current when no voltage is being applied.

    Parameters
    ----------
    delta : :class:`npt.NDArray`
        THe values of the gauge-invariant phase difference (see :func:`delta`).
    critical_current : :class:`np.float64`
        The critical current of the Josephson junction.

    Returns
    -------
    :class:`npt.NDArray`
        The current when no voltage is being applied.
    """
    return critical_current * np.sin(delta)


def ddelta_dt(potential: npt.NDArray) -> npt.NDArray:
    return 2 * scipy.constants.e / scipy.constants.hbar * potential


def josephson_inductance(critical_current: np.float64, delta: npt.NDArray):
    return M_FLUX_QUANTUM / (2 * scipy.constants.pi * critical_current * np.cos(delta))


if __name__ == "__main__":
    pass
