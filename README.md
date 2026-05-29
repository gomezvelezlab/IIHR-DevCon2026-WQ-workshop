# Developing a physics-based modeling framework to predict water quality from hillslope to watershed scales
**CIROH Developer's Conference 2026**

Conceptualization and implementation of a physics-based modeling framework to predict water quality from hillslope to watershed scales.

---

## Workshop notebooks

| Notebook | Topic |
|----------|-------|
| `Activity_1_model_exploration.ipynb` | Exploration of the Hydrology an Nitrogen models |
| `Activity_2a_remove_adsorption.ipynb` | Exploration of a Nitrogen scenario without adsorption |
| `Activity_2b_remove_tile_drainage.ipynb` | Exploration of a Hydrology scenario without tile drainage |
| `Activity_3_change_application.ipynb` | Exploration of alternative fertilizer application strategy |
| `Compare_activities_1_3.ipynb` | Comparison of results |

---

## Repo structure

```text
IIHR-DevCon2026-WQ-workshop/
├── src/devcon2026/
│   ├── hydrology/ # Hydrology submodule
│   └── nitrogen/ # Nitrogen submodule
├── notebooks/
│   ├── Activity_1_model_exploration.ipynb
│   ├── Activity_2a_remove_adsorption.ipynb
│   ├── Activity_2b_remove_tile_drainage.ipynb
│   ├── Activity_3_change_application.ipynb
│   └── Compare_activities_1_3.ipynb
└── data/
    ├── hydrology_forcings.parquet  # AORC
    ├── nitrogen_forcing_fallspring.parquet # N inputs (fertilizer strategy: Fall dominant, Spring starter)
    └── nitrogen_forcing_spring.parquet  (fertilizer strategy: Spring)
```

---

## Setup

### 2i2c (During workshop only)

1. From a browser, login to [2i2c](https://workshop.ciroh.awi.2i2c.cloud). Passwords will be provided.
2. Launch a **Medium** server (~14 GB RAM, ~4 CPUs) with image **Hillslope water Qulity Prediction**.
3. Open a new terminal and clone this repository;

    ```bash
    git clone https://github.com/gomezvelezlab/IIHR-DevCon2026-WQ-workshop.git
    ```

4. Install Python dependencies;

    ```bash
    pip install -e IIHR-DevCon2026-WQ-workshop/
    ```

5. You should now be able to start using notebooks **Activity_1_model_exploration.ipynb** provided in `notebooks/`.



