from pathlib import Path

import pandas as pd

from devcon2026.hydrology import Hydrology
from devcon2026.hydrology import HydrologyArtifactNames
from devcon2026.hydrology import HydrologyParameters
from devcon2026.hydrology import load_forcing_data
from devcon2026.nitrogen import Nitrogen
from devcon2026.nitrogen import NitrogenParameters
from devcon2026.nitrogen import NitrogenStates
from devcon2026.tables import read_table


def write_raw_hydrology_forcing(path: Path) -> None:
    pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=3, freq="h"),
            "spatial_ref": [0, 0, 0],
            "APCP_surface": [0.1, 0.0, 0.2],
            "DLWRF_surface": [300.0, 301.0, 302.0],
            "DSWRF_surface": [0.0, 100.0, 0.0],
            "PRES_surface": [99000.0, 99050.0, 99100.0],
            "SPFH_2maboveground": [0.003, 0.0031, 0.0032],
            "TMP_2maboveground": [273.15, 274.15, 275.15],
            "UGRD_10maboveground": [1.0, 1.5, 2.0],
            "VGRD_10maboveground": [0.0, 0.0, 0.0],
        }
    ).to_parquet(path, engine="fastparquet", index=False)


def test_load_forcing_data_reads_parquet(tmp_path) -> None:
    forcing_path = tmp_path / "hydrology_forcings.parquet"
    write_raw_hydrology_forcing(forcing_path)

    forcing = load_forcing_data(
        forcing_path,
        start_time="2020-01-01T01:00:00",
        end_time="2020-01-01T03:00:00",
        params=HydrologyParameters(),
    )

    assert len(forcing) == 2
    assert forcing["time"].iloc[0] == pd.Timestamp("2020-01-01T01:00:00Z")
    assert forcing["precipitation_mm"].tolist() == [0.0, 0.2]
    assert "ref_et_mm_hr" in forcing


def test_hydrology_facade_solves_exports_and_loads(tmp_path) -> None:
    output_dir = tmp_path / "hydrology"
    artifacts = HydrologyArtifactNames(
        discharge="discharge123.parquet",
        states="states123.parquet",
        fluxes="fluxes123.parquet",
        forcing="forcing123.parquet",
    )
    hydrology = Hydrology()
    hydrology.config(output_dir=output_dir, artifact_names=artifacts, hours=3)

    hydrology.solve(progress=False)
    paths = hydrology.export()
    states, fluxes, forcing = hydrology.load_outputs()

    assert hydrology.source == "generated from synthetic forcing"
    assert set(paths) == {"discharge", "states", "fluxes", "forcing"}
    assert paths["discharge"].name == "discharge123.parquet"
    assert paths["states"].name == "states123.parquet"
    assert len(states) == 3
    assert len(fluxes) == 3
    assert len(forcing) == 3

    cached = Hydrology()
    cached.config(output_dir=output_dir, artifact_names=artifacts, hours=3)
    cached.solve(progress=False)

    assert cached.source == "loaded from existing artifacts"


def test_hydrology_facade_loads_parquet_forcing(tmp_path) -> None:
    forcing_path = tmp_path / "hydrology_forcings.parquet"
    output_dir = tmp_path / "hydrology"
    write_raw_hydrology_forcing(forcing_path)

    hydrology = Hydrology(output_dir=output_dir, forcing_path=forcing_path)
    hydrology.solve(progress=False)
    hydrology.export()
    states, fluxes, forcing = hydrology.load_outputs()

    assert hydrology.source == f"generated from {forcing_path}"
    assert len(states) == 3
    assert len(fluxes) == 3
    assert len(forcing) == 3
    assert "ref_et_mm_hr" in forcing


def test_nitrogen_source_forcings_are_partitioned_by_parameters(tmp_path) -> None:
    source_path = tmp_path / "nitrogen_forcings.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1, freq="D"),
            "fertilizer_kgN_km2_day": [10.0],
            "manure_kgN_km2_day": [20.0],
            "deposition_kgN_km2_day": [30.0],
        }
    ).to_parquet(source_path, engine="fastparquet", index=False)
    base = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=2, freq="h"),
            "doy": [1.0, 1.0 + 1.0 / 24.0],
            "temp": [5.0, 5.0],
            "s": [100.0, 100.0],
        }
    )
    nitrogen = Nitrogen(
        params=NitrogenParameters(
            deposition_din_fraction=0.5,
            deposition_don_fraction=0.5,
            fertilizer_din_fraction=0.25,
            fertilizer_fon_fraction=0.75,
            manure_son_fraction=1.0,
            manure_fon_fraction=0.0,
        )
    )

    forcings = nitrogen.add_nitrogen_source_forcings(base, source_path)

    assert forcings["source_din"].tolist() == [17.5, 17.5]
    assert forcings["source_don"].tolist() == [15.0, 15.0]
    assert forcings["source_son"].tolist() == [20.0, 20.0]
    assert forcings["source_fon"].tolist() == [7.5, 7.5]


def test_nitrogen_facade_loads_hydrology_solves_and_exports(tmp_path) -> None:
    hydrology_dir = tmp_path / "hydrology"
    nitrogen_dir = tmp_path / "nitrogen"
    artifacts = HydrologyArtifactNames(
        discharge="discharge123.parquet",
        states="states123.parquet",
        fluxes="fluxes123.parquet",
        forcing="forcing123.parquet",
    )
    hydrology = Hydrology()
    hydrology.config(output_dir=hydrology_dir, artifact_names=artifacts, hours=4)
    hydrology.solve(progress=False)
    hydrology.export()

    nitrogen = Nitrogen()
    nitrogen.config(output_dir=nitrogen_dir)
    nitrogen.load_hydrology(hydrology_dir, artifact_names=artifacts)
    nitrogen.solve(progress=False)
    paths = nitrogen.export()

    assert nitrogen.df_forcings is not None
    assert nitrogen.solution_ads is not None
    assert nitrogen.solution_no_ads is not None
    assert nitrogen.mass_fluxes is not None
    assert len(nitrogen.df_forcings) == 4
    assert set(paths) == {
        "solution_with_adsorption",
        "solution_without_adsorption",
        "mass_fluxes",
    }
    assert read_table(paths["mass_fluxes"]).columns[0] == "time"


def test_nitrogen_facade_accepts_named_parameters_and_initial_states(tmp_path) -> None:
    hydrology_dir = tmp_path / "hydrology"
    artifacts = HydrologyArtifactNames()
    hydrology = Hydrology(output_dir=hydrology_dir, artifact_names=artifacts, hours=4)
    hydrology.solve(progress=False)
    hydrology.export()

    nitrogen = Nitrogen(
        output_dir=tmp_path / "nitrogen",
        params=NitrogenParameters(v_denit=0.01),
        initial_states=NitrogenStates(
            m_don=500.0,
            m_din=2500.0,
            m_son=4.5e5,
            m_fon=1.0e4,
            m_don_ads=0.0,
        ),
    )
    nitrogen.load_hydrology(hydrology_dir, artifact_names=artifacts)
    nitrogen.solve(progress=False)

    assert nitrogen.params.v_denit == 0.01
    assert nitrogen.initial_states is not None
    assert nitrogen.solution_ads is not None
    assert nitrogen.solution_ads["m_din"].iloc[0] == 2500.0
