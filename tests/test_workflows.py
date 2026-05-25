import pandas as pd

from devcon2026.hydrology import Hydrology
from devcon2026.hydrology import HydrologyArtifactNames
from devcon2026.nitrogen import Nitrogen


def test_hydrology_facade_solves_exports_and_loads(tmp_path) -> None:
    output_dir = tmp_path / "hydrology"
    artifacts = HydrologyArtifactNames(
        discharge="discharge123.csv",
        states="states123.csv",
        fluxes="fluxes123.csv",
        forcing="forcing123.csv",
    )
    hydrology = Hydrology()
    hydrology.config(output_dir=output_dir, artifact_names=artifacts, hours=3)

    hydrology.solve(progress=False)
    paths = hydrology.export()
    states, fluxes, forcing = hydrology.load_outputs()

    assert hydrology.source == "generated from synthetic forcing"
    assert set(paths) == {"discharge", "states", "fluxes", "forcing"}
    assert paths["discharge"].name == "discharge123.csv"
    assert paths["states"].name == "states123.csv"
    assert len(states) == 3
    assert len(fluxes) == 3
    assert len(forcing) == 3

    cached = Hydrology()
    cached.config(output_dir=output_dir, artifact_names=artifacts, hours=3)
    cached.solve(progress=False)

    assert cached.source == "loaded from existing CSVs"


def test_nitrogen_facade_loads_hydrology_solves_and_exports(tmp_path) -> None:
    hydrology_dir = tmp_path / "hydrology"
    nitrogen_dir = tmp_path / "nitrogen"
    artifacts = HydrologyArtifactNames(
        discharge="discharge123.csv",
        states="states123.csv",
        fluxes="fluxes123.csv",
        forcing="forcing123.csv",
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
    assert pd.read_csv(paths["mass_fluxes"]).columns[0] == "time"
