import numpy as np
import pytest

from devcon2026.nitrogen import (
    D_DIN,
    Params,
    U_DIN,
    concfactor,
    derivatives,
    exponential_moisturefactor,
    moisturefactor,
    tempfactor,
)


def test_tempfactor_thresholds() -> None:
    assert tempfactor(-1.0) == 0.0
    assert tempfactor(0.0) == 0.0
    assert tempfactor(20.0) == 1.0
    assert tempfactor(5.0) == pytest.approx(0.3535533905932738)


def test_concfactor_half_saturation() -> None:
    assert concfactor(1.5, 1.5) == 0.5


def test_moisture_factors() -> None:
    params = Params()

    assert moisturefactor(params.S_wp, params) == 0.0
    assert moisturefactor(50.0, params) == pytest.approx(0.9111111111111111)
    assert moisturefactor(params.S_s_max, params) == params.smf_sat

    assert exponential_moisturefactor(50.0, params) == 0.0
    assert exponential_moisturefactor(85.0, params) == pytest.approx(
        0.1767766952966369
    )
    assert exponential_moisturefactor(150.0, params) == 1.0


def test_denitrification_and_uptake() -> None:
    params = Params()

    assert D_DIN(0.5, 100.0, 20.0, params) == pytest.approx(0.625)
    assert U_DIN(0.5, 100.0, params) == pytest.approx(10.0)


def test_derivatives_match_current_model() -> None:
    params = Params()
    masses = np.array([1000.0, 100.0, 10.0, 50.0])

    result = derivatives(masses, S_s=100.0, temp=20.0, params=params)

    np.testing.assert_allclose(result, np.array([-8.8, -1.2, 8.04, -9.025]))
