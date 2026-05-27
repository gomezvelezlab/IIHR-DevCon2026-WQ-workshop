import numpy as np
import pandas as pd

from devcon2026.hydrology import HydrologyDerivatives
from devcon2026.hydrology import HydrologyFluxes
from devcon2026.hydrology import HydrologyForcings
from devcon2026.hydrology import HydrologyParameters
from devcon2026.hydrology import HydrologySimulationResult
from devcon2026.hydrology import HydrologyStates
from devcon2026.hydrology.export import convert_fluxes_to_nitrogen_units
from devcon2026.hydrology.export import convert_states_to_nitrogen_units
from devcon2026.hydrology.export import export_nitrogen_hydrology_inputs
from devcon2026.hydrology.physics import compute_fluxes


def test_compute_fluxes_has_nitrogen_demo_columns() -> None:
    params = HydrologyParameters()
    states = HydrologyStates(s_sn=0.01, s_s=0.04, s_gwa=0.2, s_gwp=0.5)
    forcings = HydrologyForcings(p_t=1e-7, t=5.0, e_p=2e-8)

    fluxes = compute_fluxes(0.0, states, params, forcings)

    assert set(fluxes.__dataclass_fields__) == {
        "p_sn",
        "f_sm",
        "p_r",
        "e_a",
        "q_sc",
        "q_sgwa",
        "q_gwatd",
        "q_gwac",
        "q_gwap",
        "q_gwpc",
    }


def test_prefixed_hydrology_types_are_public_api() -> None:
    assert HydrologyParameters().__class__.__name__ == "HydrologyParameters"
    assert HydrologyStates(s_sn=0.01, s_s=0.04, s_gwa=0.2, s_gwp=0.5).__class__.__name__ == "HydrologyStates"
    assert HydrologyFluxes(
        p_sn=0.0,
        f_sm=0.0,
        p_r=0.0,
        e_a=0.0,
        q_sc=0.0,
        q_sgwa=0.0,
        q_gwatd=0.0,
        q_gwac=0.0,
        q_gwap=0.0,
        q_gwpc=0.0,
    ).compute_derivatives().__class__ is HydrologyDerivatives


def test_export_nitrogen_hydrology_inputs(tmp_path) -> None:
    time = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    states = pd.DataFrame(
        {
            "s_sn": [0.01, 0.02],
            "s_s": [0.03, 0.04],
            "s_gwa": [0.5, 0.6],
            "s_gwp": [1.0, 1.1],
        },
        index=time,
    )
    fluxes = pd.DataFrame(
        {
            "p_sn": [0.0, 0.0],
            "f_sm": [0.0, 0.0],
            "p_r": [1e-8, 2e-8],
            "e_a": [1e-9, 1e-9],
            "q_sc": [1e-10, 2e-10],
            "q_sgwa": [3e-10, 4e-10],
            "q_gwatd": [5e-10, 6e-10],
            "q_gwac": [7e-10, 8e-10],
            "q_gwap": [9e-10, 1e-9],
            "q_gwpc": [2e-9, 3e-9],
        },
        index=time,
    )
    result = HydrologySimulationResult(
        discharge_cms=pd.Series([1.0, 2.0], index=time, name="discharge_cms"),
        states=states,
        fluxes=fluxes,
    )
    forcing = pd.DataFrame(
        {
            "time": time,
            "TMP_2maboveground": [273.15, 274.15],
            "APCP_surface": [0.1, 0.0],
        }
    )

    paths = export_nitrogen_hydrology_inputs(result, forcing, tmp_path)

    assert set(paths) == {"discharge", "states", "fluxes", "forcing"}
    assert pd.read_csv(paths["states"]).columns.tolist() == [
        "time",
        "s_sn",
        "s_s",
        "s_gwa",
        "s_gwp",
    ]
    assert pd.read_csv(paths["fluxes"]).columns.tolist() == [
        "time",
        "p_sn",
        "f_sm",
        "p_r",
        "e_a",
        "q_sc",
        "q_sgwa",
        "q_gwatd",
        "q_gwac",
        "q_gwap",
        "q_gwpc",
    ]


def test_export_nitrogen_hydrology_inputs_keeps_prefix_option(tmp_path) -> None:
    time = pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC")
    result = HydrologySimulationResult(
        discharge_cms=pd.Series([1.0], index=time, name="discharge_cms"),
        states=pd.DataFrame(
            {"s_sn": [0.01], "s_s": [0.03], "s_gwa": [0.5], "s_gwp": [1.0]},
            index=time,
        ),
        fluxes=pd.DataFrame(
            {
                "p_sn": [0.0],
                "f_sm": [0.0],
                "p_r": [1e-8],
                "e_a": [1e-9],
                "q_sc": [1e-10],
                "q_sgwa": [3e-10],
                "q_gwatd": [5e-10],
                "q_gwac": [7e-10],
                "q_gwap": [9e-10],
                "q_gwpc": [2e-9],
            },
            index=time,
        ),
    )
    forcing = pd.DataFrame({"time": time, "TMP_2maboveground": [273.15]})

    paths = export_nitrogen_hydrology_inputs(result, forcing, tmp_path, prefix="123")

    assert paths["discharge"].name == "discharge123.csv"
    assert paths["states"].name == "states123.csv"
    assert paths["fluxes"].name == "fluxes123.csv"


def test_nitrogen_unit_conversions() -> None:
    states = pd.DataFrame({"s_sn": [0.001], "s_s": [0.002], "s_gwa": [0.003], "s_gwp": [0.004]})
    fluxes = pd.DataFrame(
        {
            "p_sn": np.ones(1),
            "f_sm": np.ones(1),
            "p_r": np.ones(1),
            "e_a": np.ones(1),
            "q_sc": np.ones(1),
            "q_sgwa": np.ones(1),
            "q_gwatd": np.ones(1),
            "q_gwac": np.ones(1),
            "q_gwap": np.ones(1),
            "q_gwpc": np.ones(1),
        }
    )

    assert convert_states_to_nitrogen_units(states)["s_s"].iloc[0] == 2.0
    assert convert_fluxes_to_nitrogen_units(fluxes)["p_r"].iloc[0] == 86_400_000.0
