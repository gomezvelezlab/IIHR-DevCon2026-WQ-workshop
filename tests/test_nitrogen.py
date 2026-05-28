from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from devcon2026.nitrogen import NitrogenModel_SingleCV
from devcon2026.nitrogen import NitrogenParameters
from devcon2026.nitrogen import NitrogenStates
from devcon2026.nitrogen import default_soil_parameters


def test_environmental_factors() -> None:
    model = NitrogenModel_SingleCV(default_soil_parameters())

    assert model.tempfactor(-1.0) == 0.0
    assert model.tempfactor(0.0) == 0.0
    assert model.tempfactor(20.0) == 1.0
    assert model.tempfactor(5.0) == pytest.approx(0.3535533905932738)
    assert model.concfactor(1.5, 1.5) == 0.5


def test_soil_moisture_factors() -> None:
    params = default_soil_parameters()
    model = NitrogenModel_SingleCV(params)

    assert model.moisturefactor(params["s_wp"], params) == 0.0
    assert model.moisturefactor(50.0, params) == pytest.approx(0.9467903748962812)
    assert model.moisturefactor(params["s_max"], params) == params["smf_sat"]

    assert model.exponential_moisturefactor(50.0, params) == 0.0
    assert model.exponential_moisturefactor(params["s_max"] * 0.85, params) == pytest.approx(
        0.1767766952966369
    )
    assert model.exponential_moisturefactor(params["s_max"] * 1.5, params) == 1.0


def test_denitrification_and_uptake() -> None:
    params = default_soil_parameters()
    model = NitrogenModel_SingleCV(params)

    assert model.d_din(50.0, params["s_max"], 20.0, params) == pytest.approx(
        0.4613397305775973
    )
    assert model.d_din(50.0, 0.0, 20.0, params) == 0.0
    assert model.u_din(50.0, 100.0, params) == pytest.approx(10.0)


def test_dataframe_simulation_smoke() -> None:
    params = default_soil_parameters()
    model = NitrogenModel_SingleCV(params)
    time = pd.date_range("2020-01-01", periods=4, freq="h")
    forcings = pd.DataFrame(
        {
            "time": time,
            "doy": time.dayofyear + time.hour / 24.0,
            "temp": [10.0, 10.0, 11.0, 11.0],
            "s": [100.0, 101.0, 99.0, 100.0],
            "q_in_1": [1.0, 0.5, 0.0, 1.0],
            "q_in_2": [0.0, 0.5, 1.0, 0.0],
            "q_out_1": [0.2, 0.2, 0.3, 0.2],
            "q_out_2": [0.1, 0.1, 0.1, 0.1],
            "c_din_in_0": [1.0, 1.0, 1.0, 1.0],
            "c_din_in_1": [0.5, 0.5, 0.5, 0.5],
            "c_don_in_0": [0.0, 0.0, 0.0, 0.0],
            "c_don_in_1": [0.0, 0.0, 0.0, 0.0],
        }
    )
    initial_masses = np.array([500.0, 2500.0, 4.5e5, 1.0e4, 0.0])

    result = model.simulate_nitrogen_dynamics(
        df_forcings=forcings,
        M0=initial_masses,
        with_DON_ads=True,
        progress=False,
    )
    fluxes = model.get_mass_fluxes_all_species(
        M=result[["m_don", "m_din", "m_son", "m_fon"]].values,
        df_forcings=forcings,
    )

    assert list(result.columns) == [
        "time",
        "doy",
        "m_don",
        "m_din",
        "m_son",
        "m_fon",
        "m_don_ads",
        "delta_m_don",
        "s",
        "saturation_frac",
        "temp",
        "source_din",
        "source_don",
        "source_son",
        "source_fon",
        "q_total_in",
        "q_total_out",
        "c_din",
        "c_don",
        "m_din_total_flux_in",
        "m_don_total_flux_in",
        "m_din_total_flux_out",
        "m_don_total_flux_out",
    ]
    assert len(result) == len(forcings)
    assert {"r_don_flux", "q_adv_din_in_flux", "q_adv_don_out_flux", "q_source_din_flux"} <= set(
        fluxes.columns
    )
    assert np.isfinite(result[["m_don", "m_din", "m_son", "m_fon"]].values).all()


def test_dataframe_simulation_handles_dry_storage() -> None:
    params = default_soil_parameters()
    model = NitrogenModel_SingleCV(params)
    time = pd.date_range("2020-01-01", periods=4, freq="h")
    forcings = pd.DataFrame(
        {
            "time": time,
            "doy": time.dayofyear + time.hour / 24.0,
            "temp": [10.0, 10.0, 11.0, 11.0],
            "s": [100.0, 0.0, 0.0, 100.0],
            "q_in_1": [1.0, 0.0, 0.0, 1.0],
            "q_in_2": [0.0, 0.0, 0.0, 0.0],
            "q_out_1": [0.2, 0.0, 0.0, 0.2],
            "q_out_2": [0.1, 0.0, 0.0, 0.1],
            "c_din_in_0": [1.0, 1.0, 1.0, 1.0],
            "c_din_in_1": [0.5, 0.5, 0.5, 0.5],
            "c_don_in_0": [0.0, 0.0, 0.0, 0.0],
            "c_don_in_1": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = model.simulate_nitrogen_dynamics(
        df_forcings=forcings,
        M0=np.array([500.0, 2500.0, 4.5e5, 1.0e4, 0.0]),
        with_DON_ads=True,
        progress=False,
    )

    assert np.isfinite(result[["m_don", "m_din", "m_son", "m_fon"]].values).all()
    assert result.loc[result["s"] == 0.0, ["c_din", "c_don"]].eq(0.0).all().all()


def test_parameter_and_state_dataclasses_drive_model() -> None:
    params = NitrogenParameters(v_denit=0.01)
    states = NitrogenStates(
        m_don=500.0,
        m_din=2500.0,
        m_son=4.5e5,
        m_fon=1.0e4,
        m_don_ads=0.0,
    )
    model = NitrogenModel_SingleCV(params)

    assert model.params["v_denit"] == 0.01
    assert states.to_array().tolist() == [500.0, 2500.0, 4.5e5, 1.0e4, 0.0]
    assert NitrogenStates.from_array(states.to_array()) == states


def test_nitrogen_dataclasses_expose_units_and_descriptions() -> None:
    for field in fields(NitrogenParameters):
        assert field.metadata["unit"]
        assert field.metadata["description"]

    for field in fields(NitrogenStates):
        assert field.metadata["unit"] == "kg N/km2"
        assert field.metadata["description"]
