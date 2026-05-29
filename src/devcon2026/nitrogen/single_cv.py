# -----------------------------------------------------------------------------
# Author: Jesus Gomez-Velez (University of Iowa)
# Contact: jesus-gomezvelez@uiowa.edu
# Created: 2026-02-15
# Description: HYPE model.
# -----------------------------------------------------------------------------

"""
This module implements the nitrogen soil processes of the HYPE model, including denitrification and plant uptake. 
The functions are based on the mathematical descriptions of the HYPE model and are designed to be modular and reusable. 
COncentrations of dissolved constituents are in mg/L, soil pools are in kg/km2, and soil water is in mm. Recal that 
1 mm*mg/(L*day) = 1 kg/(km2*day). 

Here is an example of the parameters used for a single soil layer:

params = {
    's_wp': 20.0, # Wilting point storage (mm)
    's_max': 100.0, # Maximum soil storage (mm) -- Assumed to equal the thickness of soil layer (m) (thickm in HYPE)
    'min_dissolved_storage': 0.1, # Minimum storage for dissolved concentration and advective fluxes (mm)
    'smf_sat': 0.8, # Saturated moisture factor (satact)
    'beta_sm': 1.0, # Exponent of moisture factor (thetapow)
    'rel_saturation_low': 0.2, # Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
    'rel_saturation_high': 0.9, # High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
    'rel_sat_limit_exp': 0.7, # Limitation parameter of moisture factor (dimensionless)
    'beta_exp': 2.5, # Exponent of exponential moisture factor
    'v_degrad_son': 0.001, # Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degrhNpar in HYPE.
    'v_dissol_son': 0.01, # Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolhNpar in HYPE.
    'v_dissol_fon': 0.005, # Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolfNpar in HYPE.
    'v_min_fon': 0.02, # Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
    'v_denit': 0.05, # Maximum denitrification rate (1/day)
    'k_denit': 1.5, # Half-saturation parameter (mg/L)
    'uptake_demand': 10.0, # Plant uptake demand (kg/km2/d)
    'delta_time_solver': 1.0 # Time step (day)
}

"""

from collections.abc import Iterable
from typing import Any
from typing import Mapping
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from .types import NitrogenParameters, coerce_nitrogen_parameters

__all__ = ["NitrogenSoilLayer", "NitrogenModel_SingleCV"]


def _dry_storage_threshold(params: Mapping[str, float]) -> float:
    return params["min_dissolved_storage"]


def _dissolved_concentration(
    mass: float,
    storage: float,
    params: Mapping[str, float],
) -> float:
    return mass / storage if storage > _dry_storage_threshold(params) else 0.0


class NitrogenSoilLayer:
    """
    A class to encapsulate the HYPE nitrogen soil processes model.
    Provides methods for calculating nitrogen transformations and plant uptake.
    """
    
    def __init__(
        self,
        params: NitrogenParameters | Mapping[str, float] | None = None,
    ):
        """
        Initialize the nitrogen model with parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        self.params = coerce_nitrogen_parameters(params).to_dict()

    def set_parameters(self, params: NitrogenParameters | Mapping[str, float]): 
        """
        Receives a parameter object or dictionary and updates the existing settings.
        """
        if isinstance(params, NitrogenParameters):
            self.params = params.to_dict()
            return
        if not isinstance(params, Mapping):
            raise ValueError("Parameters should be in a mapping or NitrogenParameters.")
        self.params.update(params)


    def tempfactor(self, temp: float) -> float:
        """
        This is gamma_T in the mathematical description of our version of the HYPE model.
        Calculate a temperature factor based on Q10=2
        Reference rate at 20°C with thresholds at 0°C and 5°C.
        
        Args:
            temp: Temperature (°C)
            
        Returns:
            Temperature factor (dimensionless). 
            Monotonically increasing function with a value of 2.73 @ T=20°C.
        """
        if temp < 0.0:
            return 0.0
        
        tempf = 2.0 ** ((temp - 20.0) / 10.0)
        
        if temp < 5.0:
            tempf *= temp / 5.0
        
        return tempf

    def moisturefactor(self,
        s: float,
        params: dict,
    ) -> float:
        """
        Calculate a soil moisture dependence factor.
        
        Args:
            s: Soil storage (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm) -- Assumed to equal the thickness of soil layer (m) (thickm in HYPE)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            
        Returns:
            Soil moisture dependence factor (dimensionless, 0-1)
        """

        if s <= params['s_wp']:
            return 0.0
        if s >= params['s_max']:
            return params['smf_sat']

        term1 = (1.0 - params['smf_sat']) * ((params['s_max'] - s) / (params['rel_saturation_high'] * params['s_max'])) ** params['beta_sm'] + params['smf_sat']
        term2 = ((s - params['s_wp']) / (params['rel_saturation_low'] * params['s_max'])) ** params['beta_sm']
        return min(1.0, term1, term2)

    def concfactor(self, c: float, k: float) -> float:
        """
        Calculate a concentration dependence factor based on half saturation function.
        This is a Michaelis-Menten Kinetics function. Equation: $f = c /(c + K}$ with 
        $c$ being the concentration and $K$ the half-saturation parameter (or Michaelis constant). 
        The constant $K$ represents the concentration at which the f is exactly $1/2$.
        
        Args:
            c: Concentration (mg/L)
            k: Half-saturation parameter (mg/L)
            
        Returns:
            Concentration factor (dimensionless, 0-1)
        """

        return c / (c + k)

    def exponential_moisturefactor(self,
        s: float,
        params: dict,
    ) -> float:
        """
        Calculate an exponential soil moisture dependence factor for denitrification.
        
        Args:   
            s: Soil storage (mm)
            s_max: Maximum soil storage
            rel_sat_limit_exp: Limitation parameter of moisture factor (dimensionless)
            beta_exp: Exponent of exponential moisture factor
            
        Returns:
            Exponential moisture factor (dimensionless)
        """
        rel_sat = s / params['s_max']

        if rel_sat > 1.0:
            return 1.0
        
        if rel_sat > params['rel_sat_limit_exp']:
            return (((rel_sat - params['rel_sat_limit_exp']) / (1.0 - params['rel_sat_limit_exp'])) ** params['beta_exp'])
        
        return 0.0


    def get_adsorption_of_don(self,
        m_don: float,
        s: float,
        params: dict,
    ) -> float:
        """
        Calculate the adsorption rate of dissolved organic nitrogen (DON) using a Freundlich isotherm. 
        This process regulates the distribution of nitrogen between the dissolved phase (water) and 
        the adsorbed phase (soil particles), effectively acting as a buffer that slows down the leaching 
        of organic nitrogen. The model assumes $C_{ads} = K_{F} * C_{don}^{1/n}$ where $C_{ads}$ is the 
        concentration of adsorbed DON (mg-N/kg-soil), $K_{F}$ is the Freundlich isotherm empirical constant 
        (`freundlich_constant`; units of (mg-N/kg-soil)/(mg-N/L)^n), $n$ is the Freundlich exponent 
        (`freundlich_exponent`; [0, 1]; dimensionless), and $C_{don}$ is the concentration of dissolved organic nitrogen (mg-N/L). 
        
        Key considerations for the `freundlich_constant`:
            - Soil Dependent: This parameter is typically defined by soil type.
            - Retention Capacity: A higher `freundlich_constant` value means the soil 
                has a higher affinity for organic nitrogen, leading to more nitrogen being "stuck" to soil particles 
                and less being available for transport through tile drains or deep percolation.
            - Clay/Organic Content: In practice, `freundlich_constant` is usually set higher for clay-rich soils or 
                soils with high organic matter, as these have more surface area and binding sites for DON.

        Vitek, R.; Masini, J. C. Nonlinear Regression for Treating Adsorption Isotherm Data to Characterize New Sorbents: Advantages over Linearization Demonstrated with Simulated and Experimental Data. Heliyon 2023, 9 (4), e15128. https://doi.org/10.1016/j.heliyon.2023.e15128.


        For `freundlich_exponent = 1`, the values of `freundlich_constant` in mg-N/kg-soil are typically:

                Typical Range: 150 to 10,400 (depending on soil type).

                Soil Texture Classfreuc Value Range -- Adsorption Affinity
                Sandy Soils: 1 – 50 -- Low: Low adsorption; high leaching risk.
                Loamy Soils: 50 – 200 -- Moderate: Moderate adsorption; buffers DON flow.
                Clayey Soils: 200 – >10000 -- High: High adsorption; strongly retains organic N.

                In the HYPE model (written in Fortran), the adsorption and desorption of dissolved organic nitrogen (DON) 
                are implemented as a mass-balance equilibrium adjustment within each soil layer at every timestep.

        Args:
            m_don: Mass of soil dissolved organic nitrogen (kg N/km2)
            s: Soil water (mm)
            params: Dictionary of model parameters

        Returns:
            Mass of adsorbed organic nitrogen (kg N/km2)
        """
        c_don = m_don / s if s > 0 else 0.0 # Concentration of DON in mg/L
        c_don_ads = params['freundlich_constant'] * (c_don) ** params['freundlich_exponent'] # Concentration of adsorbed DON (mg-N/kg-soil)
        m_don_ads = s * params['soil_bulk_density'] * c_don_ads # Mass of adsorbed DON (kg N/km2)

        return m_don_ads

    def get_adsorption_of_don_inv(self,
        m_don_ads: float,
        s: float,
        params: dict,
    ) -> float:
        """
        Calculate the mass of dissolved organic nitrogen (DON) that will be present under equilibrium adsorption conditions using a Freundlich isotherm. 
        
        Args:
            m_don_ads: Mass of adsorbed organic nitrogen (kg N/km2)
            s: Soil water (mm)
            params: Dictionary of model parameters

        Returns:
            m_don: Mass of soil dissolved organic nitrogen (kg N/km2) 
        """
        m_don = s * (m_don_ads/(s*params['freundlich_constant']*params['soil_bulk_density']))**(1/params['freundlich_exponent']) if s > 0 else 0.0 # Mass of dissolved DON (kg N/km2)

        return m_don
    
    def get_don_mass_balance_equilibrium_adjustment(
        self,
        m_don_current: float,
        m_don_ads_previous: float,
        s: float,
        params: dict,
    ) -> Tuple[float, float, float]:
        """
        Calculate the equilibrium adjustment for dissolved organic nitrogen (DON) mass balance due to adsorption/desorption.

        M_don_total = M_don_dissolved + M_don_adsorbed

        where M_don_dissolved = m_don = c_don * s and M_don_adsorbed = c_ads * soil_bulk_density * soil_layer_thickness
        
        Args:
            m_don_current: Current mass of soil dissolved organic nitrogen (kg N/km2)
            m_don_ads_previous: Previous mass of adsorbed organic nitrogen (kg N/km2)
            s: Soil water (mm)
            params: Dictionary of model parameters
        Returns:
            Tuple containing:

            """
        m_don_eq = self.get_adsorption_of_don_inv(m_don_ads_previous, s, params) # Mass of dissolved DON at equilibrium (kg N/km2)
        delta_m_don = m_don_current - m_don_eq # Change in dissolved DON mass to reach equilibrium (kg N/km2)

        # delta_m_don = delta_m_don if s > 0 else 0.0 # If soil water is zero, then there is no dissolved DON and no adjustment is needed, so set delta_m_don to zero to avoid NaN values.    

        m_don_new = m_don_current - delta_m_don # Adjusted mass of dissolved DON after equilibrium adjustment (kg N/km2)
        m_don_ads_new = m_don_ads_previous + delta_m_don # Adjusted mass of adsorbed DON after equilibrium adjustment (kg N/km2)

        return m_don_new, m_don_ads_new, delta_m_don


    def d_din(self,
        m_din: float, 
        s: float, 
        temp: float, 
        params: dict,
    ) -> float:
        """
        Denitrification of inorganic nitrogen in one soil layer.
        HYPE name: soil_denitrification
        
        Args:
            m_din: Mass of soil dissolved inorganic nitrogen (kg N/km2)
            s: Soil water (mm)
            s_max: Maximum water content / pore volume (mm)
            temp: Soil temperature (°C)        
            v_denit: Maximum denitrification rate (1/day)
            k_denit: Half-saturation parameter (mg/L)
            rel_sat_limit_exp: Limitation parameter of moisture factor (dimensionless)
            beta_exp: Exponent of exponential soil moisture factor
            
        Returns:
            Denitrification rate (kg N/km2/d)
        """

        if s <= _dry_storage_threshold(params):
            return 0.0

        c_din = m_din / s # Inorganic N concentration (mg/L)

        # Environmental factors (HYPE: tmpfcn, smfcn, concfcn)
        tmpf = self.tempfactor(temp)
        smf = self.exponential_moisturefactor(s, params)
        # print(c_din, params['k_denit'], s)
        concf = self.concfactor(c_din, params['k_denit'])
        
        # Denitrification (HYPE: sink)
        d_din = params['v_denit'] * m_din * tmpf * smf * concf
        
        return d_din


    def get_potential_din_demaid(self, doy: float, params: dict) -> float:

        # Example of a seasonal uptake demand with a sinusoidal pattern
        potential_demand = params['uptake_demand'] * (1 + np.sin(2*np.pi*((doy - params['phase_shift_uptake_demand']))/365))** params['exponent_uptake_demand']
        return potential_demand 



    def u_din(self,
        m_din: float, 
        s: float, 
        params: dict,
    ) -> float:
        
        """
        Calculate plant nutrient uptake from soil.
        Operates on first two soil layers only.
        
        Args:
            m_din: Mass of soil dissolved inorganic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Water content at wilting point (mm)
            uptake_demand: Plant uptake demand (kg/km2/d) 
            delta_time_solver: Time step (day)
            
        Returns:
            uptake: Plant uptake (kg N/km2/d)
        """

        # Maximum uptake fraction (limited by wilting point)
        s_avail = s - params['s_wp']
        water_corr_factor = (s_avail/ s) if ((s > 0) and (s_avail > 0)) else 0.0

        # Actual uptake (limited by demand and available pool = m_din)
        # available inorganic N pool (kg/km2)
        # actual uptake limited by demand and available pool
        max_uptake_flux = water_corr_factor * m_din / params['delta_time_solver'] # Uptake flux that will consume the available m_din during time step (kg/km2/d)
        # uptake_demand = self.get_potential_din_demaid(doy=0, params=params) # We can include the seasonal pattern of uptake demand if we want, but for now we will use a constant demand.
        uptake_demand = params['uptake_demand'] # Constant demand for now (kg/km2/d)
        
        return np.min([max_uptake_flux, uptake_demand])

    def r_degrad_son(self,
        m_son: float,
        s: float,
        temp: float, 
        params: dict,
    ) -> float:
        """
        Calculate the degradation rate of soil slow organic nitrogen (SON).
        
        Args:
            m_son: Mass of soil slow organic nitrogen (kg-N/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_degrad_son: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degrhNpar in HYPE.
            
        Returns:
            Degradation rate of soil slow organic nitrogen (kg N/km2/d)
        """
        
        # Environmental factors
        tmpf = self.tempfactor(temp)
        smf = self.moisturefactor(s, params)
        
        # Degradation rate
        reaction_rate = params['v_degrad_son'] * m_son * tmpf * smf
        
        return reaction_rate

    def r_dissol_son(self,
        m_son: float,
        s: float,
        temp: float, 
        params: dict,
    ) -> float:
        """
        Calculate the dissolution rate of soil slow organic nitrogen (SON).
        
        Args:
            m_son: Mass of soil slow organic nitrogen (kg-N/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_dissol_son: Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolhNpar in HYPE.
            
        Returns:
            Dissolution rate of soil slow organic nitrogen (kg N/km2/d)
        """
        
        # Environmental factors
        tmpf = self.tempfactor(temp)
        smf = self.moisturefactor(s, params)
        
        # Dissolution rate
        reaction_rate = params['v_dissol_son'] * m_son * tmpf * smf
        
        return reaction_rate

    def r_dissol_fon(self,
        m_fon: float,
        s: float,
        temp: float, 
        params: dict,
    ) -> float:
        """
        Calculate the dissolution rate of soil fast organic nitrogen (FON).
        
        Args:
            m_fon: Mass of soil fast organic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_dissol_fon: Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolfNpar in HYPE.
            
        Returns:
            Dissolution rate of soil fast organic nitrogen (kg N/km2/d)
        """
        
        # Environmental factors
        tmpf = self.tempfactor(temp)
        smf = self.moisturefactor(s, params)
        
        # Dissolution rate
        reaction_rate = params['v_dissol_fon'] * m_fon * tmpf * smf
        
        return reaction_rate

    def r_min_fon(self,
        m_fon: float,
        s: float,
        temp: float,
        params: dict,
    ) -> float:
        """
        Calculate the mineralization rate of soil fast organic nitrogen (FON).
        
        Args:
            m_fon: Mass of soil fast organic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_min_fon: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            
        Returns:
            Mineralization rate of soil fast organic nitrogen (kg N/km2/d)
        """
        
        # Environmental factors
        tmpf = self.tempfactor(temp)
        smf = self.moisturefactor(s, params)
        
        # Mineralization rate
        reaction_rate = params['v_min_fon'] * m_fon * tmpf * smf
        
        return reaction_rate


    def r_son(self,
        m_son: float,
        s: float,
        temp: float,
        params: dict,
    ) -> float:
        """
        Calculate the reactionrate of soil slow organic nitrogen (SON).
        
        Args:
            m_fon: Mass of soil fast organic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_degrad_son: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degradSONpar in HYPE.
            v_dissol_son: Maximum dissolution rate of soil slow organic nitrogen (1/day) -- This is the parameter dissolSONpar in HYPE.
            
        Returns:
            Reaction rate of soil slow organic nitrogen (kg N/km2/d)
        """
        
        # Mineralization rate
        reaction_rate = - self.r_degrad_son(m_son, s, temp, params) - self.r_dissol_son(m_son, s, temp, params)
        
        return reaction_rate

    def r_fon(self,
        m_fon: float,
        m_son: float,
        s: float,
        temp: float, 
        params: dict,
    ) -> float:
        """
        Calculate the reactionrate of soil fast organic nitrogen (FON).
        
        Args:
            m_fon: Mass of soil fast organic nitrogen (kg/km2)
            m_son: Mass of soil slow organic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_degrad_son: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degradSONpar in HYPE.
            v_dissol_fon: Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolFONpar in HYPE.
            v_min_fon: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            
        Returns:
            Reaction rate of soil fast organic nitrogen (kg N/km2/d)
        """

        # Mineralization rate
        reaction_rate = (
            self.r_degrad_son(m_son, s, temp, params)
            - self.r_dissol_fon(m_fon, s, temp, params)
            - self.r_min_fon(m_fon, s, temp, params)
        )

        return reaction_rate

    def r_don(self,
        m_fon: float,
        m_son: float,
        s: float,
        temp: float,
        params: dict,
    ) -> float:
        """
        Calculate the reactionrate of soil dissolved organic nitrogen (DON).
        
        Args:
            m_fon: Mass of soil fast organic nitrogen (kg/km2)
            m_son: Mass of soil slow organic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_degrad_son: Maximum degradation rate of soil slow organic nitrogen (1/day) -- This is the parameter degradSONpar in HYPE.
            v_dissol_fon: Maximum dissolution rate of soil fast organic nitrogen (1/day) -- This is the parameter dissolFONpar in HYPE.
            v_min_fon: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            
        Returns:
            Reaction rate of soil dissolved organic nitrogen (kg N/km2/d)
        """
        
        reaction_rate = self.r_dissol_son(m_son, s, temp, params)  + self.r_dissol_fon(m_fon, s, temp, params) 
        
        return reaction_rate

    def r_din(self,
        m_fon: float,
        m_din: float,
        s: float,
        temp: float,
        params: dict,
    ) -> float:
        """
        Calculate the reactionrate of soil dissolved inorganic nitrogen (DIN).
        
        Args:
            m_fon: Mass of soil fast organic nitrogen (kg/km2)
            m_son: Mass of soil slow organic nitrogen (kg/km2)
            s: Soil water (mm)
            s_wp: Wilting point storage (mm)
            s_max: Maximum soil storage (mm)
            smf_sat: Saturated moisture factor (satact)
            beta_sm: Exponent of moisture factor (thetapow)
            beta_exp: Exponent of exponential soil moisture factor for denitrification
            rel_sat_limit: Limitation parameter of moisture factor for denitrification (dimensionless)
            rel_saturation_low: Low relative saturation (dimensionless, 0-1) -- This is (thetalow, %)/100
            rel_saturation_high: High relative saturation (dimensionless, 0-1) -- This is (thetaupp, %)/100
            temp: Soil temperature (°C)
            v_min_fon: Maximum mineralization rate of soil fast organic nitrogen (1/day) -- This is the parameter minfNpar in HYPE.
            uptake_demand: Plant uptake demand (kg/km2/d)
            v_denit: Maximum denitrification rate (1/day)
            k_denit: Half-saturation concentration for denitrification (mg/L)
            delta_time_solver: Time step (day)
            
        Returns:
            Reaction rate of soil dissolved inorganic nitrogen (kg N/km2/d)
        """
        # c_din = m_din / s if s > 0 else 0.0 # Inorganic N concentration (mg/L)

        # r1 = self.r_min_fon(m_fon, s, temp, params)
        # r2 = self.u_din(m_din, s, params)
        # r3 = self.d_din(m_din, s, temp, params)

        # print(f"R_min: {r1:.2f} kg/km2/d, u_din: {r2:.2f} kg/km2/d, d_din: {r3:.2f} kg/km2/d")

        reaction_rate = self.r_min_fon(m_fon, s, temp, params) - self.u_din(m_din, s, params) - self.d_din(m_din, s, temp, params)
        
        return reaction_rate

    def Q_SON(self) -> float:
        """
        Calculate the source/sink term of soil slow organic nitrogen (SON) due to plant litter input.
        This is a placeholder function that can be expanded to include more complex dynamics of plant litter input.
        
        Returns:
            Source/sink term of soil slow organic nitrogen (kg N/km2/d)
        """
        return 0.0

    def Q_FON(self) -> float:
        """
        Calculate the source/sink term of soil fast organic nitrogen (FON) due to plant litter input, dry atmospheric deposition, and solid fertilizers and manure.
        This is a placeholder function that can be expanded to include more complex dynamics of plant litter input.
        
        Returns:
            Source/sink term of soil fast organic nitrogen (kg N/km2/d)
        """
        return 0.0

    def Q_DIN(self) -> float:
        """
        Calculate the source/sink term of soil dissolved inorganic nitrogen (DIN) due to wet atmospheric deposition and liquid fertilizers and manure.
        This is a placeholder function that can be expanded to include more complex dynamics of plant litter input.
        
        Returns:
            Source/sink term of soil dissolved inorganic nitrogen (kg N/km2/d)
        """
        return 0.0


    def get_derivatives_all_species(self,
        M: np.ndarray,
        s: float,
        q_in: np.ndarray,
        q_out: np.ndarray,
        c_don_in: np.ndarray,
        c_din_in: np.ndarray,
        temp: float,
        source_din: float = 0.0,
        source_don: float = 0.0,
        source_son: float = 0.0,
        source_fon: float = 0.0) -> np.ndarray:

        """
        Calculate the derivatives of the mass ODEs for soil dissolved organic nitrogen (DON), 
        soil dissolved inorganic nitrogen (DIN), soil slow organic nitrogen (SON), soil fast organic nitrogen (FON).
        
        Args:
            M: Array of masses [m_don, m_din, m_son, m_fon] (kg/km2)
            s: Soil water (mm)
            q_in: Array of inflow rates (mm/d)
            q_out: Array of outflow rates (mm/d)
            c_don_in: Array of inflow concentrations for dissolved organic nitrogen (mg/L)
            c_din_in: Array of inflow concentrations for dissolved inorganic nitrogen (mg/L)
            temp: Soil temperature (°C)

        Returns:
            Array of derivatives [dm_don/dt, dm_din/dt, dm_son/dt, dm_fon/dt] (kg N/km2/d)
        """

        M[0] = max(M[0], 0.0) # Ensure non-negative mass for DON
        M[1] = max(M[1], 0.0) # Ensure non-negative mass for DIN
        M[2] = max(M[2], 0.0) # Ensure non-negative mass for SON
        M[3] = max(M[3], 0.0) # Ensure non-negative mass for FON

        c_don = _dissolved_concentration(M[0], s, self.params)
        c_din = _dissolved_concentration(M[1], s, self.params)

        # print(f"c_don: {c_don:.2f} mg/L, c_din: {c_din:.2f} mg/L, s: {s:.2f} mm, temp: {temp:.2f} °C")

        # r1 = self.r_din(M[3], M[1], s, temp, self.params) 
        # r2 =  np.sum(q_in * c_din_in) 
        # r3 = np.sum(q_out * c_din)
        # print(f"r_din: {r1:.2f} kg/km2/d, q_in*c_din_in: {r2:.2f} kg/km2/d, q_out*c_din: {r3:.2f} kg/km2/d")

        assert not np.isnan(c_din), "c_din is NaN"

        dM_dt = np.array([
            self.r_don(M[3], M[2], s, temp, self.params) + source_don + np.sum(q_in * c_don_in) - np.sum(q_out * c_don), # dm_don/dt
            self.r_din(M[3], M[1], s, temp, self.params) + source_din + self.Q_DIN() + np.sum(q_in * c_din_in) - np.sum(q_out * c_din), # dm_din/dt
            self.r_son(M[2], s, temp, self.params) + source_son + self.Q_SON(),  # dm_son/dt
            self.r_fon(M[3], M[2], s, temp, self.params) + source_fon + self.Q_FON() # dm_fon/dt
            ])
        
        return dM_dt

    def get_mass_fluxes_all_species(self,
        M: np.ndarray, 
        df_forcings: pd.DataFrame) -> pd.DataFrame:

        """
        Calculate the derivatives of the mass ODEs for soil dissolved organic nitrogen (DON), 
        soil dissolved inorganic nitrogen (DIN), soil slow organic nitrogen (SON), soil fast organic nitrogen (FON).

        Args:
            M: Array of masses with dimensions (n_time_steps, 4) [m_don, m_din, m_son, m_fon] (kg/km2)
            s: Soil water (mm)
            q_in: Array of inflow rates (mm/d)
            q_out: Array of outflow rates (mm/d)
            c_don_in: Array of inflow concentrations for dissolved organic nitrogen (mg/L)
            c_din_in: Array of inflow concentrations for dissolved inorganic nitrogen (mg/L)
            temp: Soil temperature (°C)

        Returns:
            Array of derivatives [dm_don/dt, dm_din/dt, dm_son/dt, dm_fon/dt] (kg N/km2/d)
        """

        M = M.reshape((-1, 4)) # Ensure M has shape (n_time_steps, 4)

        # Unpack the forcing data for the time steps
        time_index = df_forcings["time"]
        varnames_q_in = [name for name in df_forcings.columns.values if "q_in_" in name]
        varnames_q_out = [name for name in df_forcings.columns.values if "q_out_" in name]
        varnames_c_din_in = [name for name in df_forcings.columns.values if "c_din_in_" in name]
        varnames_c_don_in = [name for name in df_forcings.columns.values if "c_don_in_" in name]

        s = df_forcings['s'].values
        q_in = df_forcings[varnames_q_in].values
        q_out = df_forcings[varnames_q_out].values
        c_din_in = df_forcings[varnames_c_din_in].values
        c_don_in = df_forcings[varnames_c_don_in].values
        temp = df_forcings['temp'].values
        source_din = df_forcings.get("source_din", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        source_don = df_forcings.get("source_don", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        source_son = df_forcings.get("source_son", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        source_fon = df_forcings.get("source_fon", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        
        # Reaction mass fluxes for DON, DIN, SON, FON (kg N/km2/d)
        r_don_flux = np.zeros(M.shape[0]) 
        r_din_flux = np.zeros(M.shape[0])
        r_son_flux = np.zeros(M.shape[0])
        r_fon_flux = np.zeros(M.shape[0])
        r_degrad_son_flux = np.zeros(M.shape[0])
        r_dissol_son_flux = np.zeros(M.shape[0])
        r_dissol_fon_flux = np.zeros(M.shape[0])
        r_min_fon_flux = np.zeros(M.shape[0])
        u_din_flux = np.zeros(M.shape[0])
        d_din_flux = np.zeros(M.shape[0])

        # Advective mass fluxes for DON and DIN (kg N/km2/d)
        q_adv_don_in_flux = np.zeros(M.shape[0])
        q_adv_don_out_flux = np.zeros(M.shape[0])
        q_adv_din_in_flux = np.zeros(M.shape[0])
        q_adv_din_out_flux = np.zeros(M.shape[0])


        # Sources of mass for DON and DIN (kg N/km2/d)
        q_source_din_flux = np.zeros(M.shape[0])
        q_source_don_flux = np.zeros(M.shape[0])
        q_source_son_flux = np.zeros(M.shape[0])
        q_source_fon_flux = np.zeros(M.shape[0])

        for i in range(M.shape[0]):

            r_don_flux[i] = self.r_don(M[i,3], M[i,2], s[i], temp[i], self.params) 
            r_din_flux[i] = self.r_din(M[i,3], M[i,1], s[i], temp[i], self.params)
            r_son_flux[i] = self.r_son(M[i,2], s[i], temp[i], self.params) 
            r_fon_flux[i] = self.r_fon(M[i,3], M[i,2], s[i], temp[i], self.params)

            r_degrad_son_flux[i] = self.r_degrad_son(M[i,2], s[i], temp[i], self.params)
            r_dissol_son_flux[i] = self.r_dissol_son(M[i,2], s[i], temp[i], self.params)
            r_dissol_fon_flux[i] = self.r_dissol_fon(M[i,3], s[i], temp[i], self.params)
            r_min_fon_flux[i] = self.r_min_fon(M[i,3], s[i], temp[i], self.params)
            u_din_flux[i] = self.u_din(M[i,1], s[i], self.params)
            d_din_flux[i] = self.d_din(M[i,1], s[i], temp[i], self.params)

            q_adv_don_in_flux[i] = np.sum(q_in[i, :] * c_don_in[i, :])
            c_don = _dissolved_concentration(M[i,0], s[i], self.params)
            c_din = _dissolved_concentration(M[i,1], s[i], self.params)
            q_adv_don_out_flux[i] = np.sum(q_out[i, :]) * c_don
            q_adv_din_in_flux[i] = np.sum(q_in[i, :] * c_din_in[i, :])
            q_adv_din_out_flux[i] = np.sum(q_out[i, :]) * c_din

            q_source_din_flux[i] = source_din[i] + self.Q_DIN()
            q_source_don_flux[i] = source_don[i]
            q_source_son_flux[i] = source_son[i] + self.Q_SON()
            q_source_fon_flux[i] = source_fon[i] + self.Q_FON()


        df_mass_fluxes = pd.DataFrame({
            'r_don_flux': r_don_flux,
            'r_din_flux': r_din_flux,
            'r_son_flux': r_son_flux,
            'r_fon_flux': r_fon_flux,
            'r_degrad_son_flux': r_degrad_son_flux,
            'r_dissol_son_flux': r_dissol_son_flux,
            'r_dissol_fon_flux': r_dissol_fon_flux,
            'r_min_fon_flux': r_min_fon_flux,
            'u_din_flux': u_din_flux,
            'd_din_flux': d_din_flux,
            'q_adv_don_in_flux': q_adv_don_in_flux,
            'q_adv_don_out_flux': q_adv_don_out_flux,
            'q_adv_din_in_flux': q_adv_din_in_flux,
            'q_adv_din_out_flux': q_adv_din_out_flux,
            'q_source_din_flux': q_source_din_flux,
            'q_source_don_flux': q_source_don_flux,
            'q_source_son_flux': q_source_son_flux,
            'q_source_fon_flux': q_source_fon_flux
        })

        return df_mass_fluxes


    def get_derivatives_dissolved_species(self,
        M: np.ndarray,
        s: float,
        q_in: np.ndarray,
        q_out: np.ndarray,
        c_don_in: np.ndarray,
        c_din_in: np.ndarray,
        temp: float) -> np.ndarray:

        """
        Calculate the derivatives of the mass ODEs for soil dissolved organic nitrogen (DON), 
        soil dissolved inorganic nitrogen (DIN)
        
        Args:
            M: Array of masses [m_don, m_din] (kg/km2)
            s: Soil water (mm)
            q_in: Array of inflow rates (mm/d)
            q_out: Array of outflow rates (mm/d)
            c_don_in: Array of inflow concentrations for dissolved organic nitrogen (mg/L)
            c_din_in: Array of inflow concentrations for dissolved inorganic nitrogen (mg/L)
            temp: Soil temperature (°C)

        Returns:
            Array of derivatives [dm_don/dt, dm_din/dt] (kg N/km2/d)
        """

        M[0] = max(M[0], 0.0) # Ensure non-negative mass for DON
        M[1] = max(M[1], 0.0) # Ensure non-negative mass for DIN

        c_don = _dissolved_concentration(M[0], s, self.params)
        c_din = _dissolved_concentration(M[1], s, self.params)

        dM_dt = np.array([
            np.sum(q_in * c_don_in) - np.sum(q_out * c_don), # dm_don/dt
            - self.d_din(M[1], s, temp, self.params) + self.Q_DIN() + np.sum(q_in * c_din_in) - np.sum(q_out * c_din) # dm_din/dt
            ])
        
        return dM_dt
    
    def simulate_nitrogen_dynamics(self,
        df_forcings: pd.DataFrame,
        M0: np.ndarray,
        with_DON_ads: bool = True,
        progress: bool = True,
        progress_desc: str | None = None
    ) -> pd.DataFrame:
        
        """Simulate the nitrogen dynamics using the ODE solver.

        Args:
            df_forcings: DataFrame containing the time series of forcing variables (soil moisture, temperature, inflow/outflow fluxes and concentrations)
            M0: Initial masses of [M_DON, M_DIN, M_SON, M_FON, M_DON_ADS] (kg/km2)
            with_DON_ads: Whether to include the adsorption dynamics of DON in the simulation (if False, the model will not simulate the dynamics of M_DON_ADS and will only return the time series of M_DON, M_DIN, M_SON, M_FON
            progress: Whether to show a progress bar during the simulation
            progress_desc: Description for the progress bar
        Returns:
            DataFrame containing the time series of simulated masses of DON, DIN, SON, FON, DON_ADS and the corresponding forcing variables.
        """

        def solve_ivp_fun(
            t: float,
            y: NDArray[Any],
            model_CV: NitrogenSoilLayer,
            forcings: dict[str, Any]
        ) -> NDArray[Any]:

            return model_CV.get_derivatives_all_species(
                M=y, # Initial masses of [M_DON, M_DIN, M_SON, M_FON] (kg/km2) -- Do not include M_DON_ADS in the state vector since we are not modeling its dynamics for now
                s=forcings['s'], # Initial soil moisture storage (mm)
                q_in=forcings['q_in'], # Inflow fluxes (mm/day) 
                q_out=forcings['q_out'], # Outflow fluxes (mm/day) 
                c_don_in=forcings['c_don_in'], # Concentration of DON in inflow (mg/L) 
                c_din_in=forcings['c_din_in'], # Concentration of DIN in inflow (mg/L) 
                temp=forcings['temp'], # Soil temperature (°C)
                source_din=forcings['source_din'],
                source_don=forcings['source_don'],
                source_son=forcings['source_son'],
                source_fon=forcings['source_fon'],
            )     

        df_forcings = df_forcings.reset_index(drop=True)
        # Unpack the forcing data for the time steps
        time_index = pd.DatetimeIndex(df_forcings["time"])
        varnames_q_in = [name for name in df_forcings.columns.values if "q_in_" in name]
        varnames_q_out = [name for name in df_forcings.columns.values if "q_out_" in name]
        varnames_c_din_in = [name for name in df_forcings.columns.values if "c_din_in_" in name]
        varnames_c_don_in = [name for name in df_forcings.columns.values if "c_don_in_" in name]

        s = df_forcings['s'].to_numpy(dtype=float)
        q_in = df_forcings[varnames_q_in].to_numpy(dtype=float)
        q_out = df_forcings[varnames_q_out].to_numpy(dtype=float)
        c_din_in = df_forcings[varnames_c_din_in].to_numpy(dtype=float)
        c_don_in = df_forcings[varnames_c_don_in].to_numpy(dtype=float)
        temp = df_forcings['temp'].to_numpy(dtype=float)
        source_din = df_forcings.get("source_din", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        source_don = df_forcings.get("source_don", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        source_son = df_forcings.get("source_son", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)
        source_fon = df_forcings.get("source_fon", pd.Series(0.0, index=df_forcings.index)).to_numpy(dtype=float)

        # Solver time step in days
        delta_time_solver_in_days = self.params['delta_time_solver']

        iter_time: Iterable[pd.Timestamp] = list(time_index[1:]) # Skip the first time step since it's used for initial conditions
        if progress:
            iter_time = tqdm(
                iter_time,
                total=len(time_index[1:]),
                desc=progress_desc or "simulate",
                unit="step",
            )

        y = M0.copy() # Initial state vector for [M_DON, M_DIN, M_SON, M_FON]
        y = y[0:4] # Only include the masses of DON, DIN, SON, FON in the state vector since we are not modeling the dynamics of M_DON_ADS for now
        y_ads = M0[4] if len(M0) > 4 else 0.0 # Initial mass of adsorbed DON (kg/km2), set to 0 if not provided
        delta_m_don = 0.0 # Initial change in mass of DON due to adsorption/desorption (kg/km2)
        states_history = [np.append(np.append(y.copy(), y_ads), delta_m_don)] # List to store the history of state variables and adsorbed DON mass at each time step

        for i, current_time in enumerate(iter_time):

            # t_start = current_time.timestamp()
            # t_end = (current_time + pd.Timedelta(days=delta_time_solver_in_days)).timestamp()

            forcings = {
                's': s[i], # Soil storage (mm)
                'q_in': q_in[i,:], # Inflow fluxes (mm/day)
                'q_out': q_out[i,:], # Outflow fluxes (mm/day) 
                'c_don_in': c_don_in[i,:], # Concentration of DON in inflow (mg/L)
                'c_din_in': c_din_in[i,:], # Concentration of DIN in inflow (mg/L) 
                'temp': temp[i], # Soil temperature (°C)
                'source_din': source_din[i],
                'source_don': source_don[i],
                'source_son': source_son[i],
                'source_fon': source_fon[i],
            }

            sol = solve_ivp(
                solve_ivp_fun,
                (0, delta_time_solver_in_days),
                y, 
                args=(self, forcings),
                method="LSODA",
            )

            if not sol.success:
                raise RuntimeError(f"Solver failed at {current_time}: {sol.message}")

            y = np.maximum(0.0, sol.y[:, -1]).astype(np.float64)

            # Adsorption dynamics for DON
            if with_DON_ads:

                m_don_new, m_don_ads_new, delta_m_don = self.get_don_mass_balance_equilibrium_adjustment(
                    m_don_current = y[0],
                    m_don_ads_previous = y_ads,
                    s=s[i],
                    params=self.params
                )

                y[0] = m_don_new
                y_ads = m_don_ads_new

            y[0] = y[0] if s[i] > 0 else 0.0 # If water mass in storage is zero, then set dissolved mass to zero to avoid NaN concentrations
            y[1] = y[1] if s[i] > 0 else 0.0 # If water mass in storage is zero, then set dissolved mass to zero to avoid NaN
        
            # Save the state variables and the adsorbed DON mass at the current time step
            states_history.append(np.append(np.append(y, y_ads), delta_m_don))

        # column names for the four state variables
        var_names = ['m_don', 'm_din', 'm_son', 'm_fon', 'm_don_ads', 'delta_m_don']

        # make a DataFrame from the array history
        df_sln = pd.DataFrame(np.array(states_history), columns=var_names)
        df_sln['time'] = time_index
        df_sln['doy'] = df_forcings['doy'].values
        df_sln = df_sln[['time', 'doy', 'm_don', 'm_din', 'm_son', 'm_fon', 'm_don_ads', 'delta_m_don']] # Reorder columns to have time and doy first
        df_sln['s'] = s
        df_sln['saturation_frac'] = s / self.params['s_max']
        df_sln['temp'] = temp
        df_sln['source_din'] = source_din
        df_sln['source_don'] = source_don
        df_sln['source_son'] = source_son
        df_sln['source_fon'] = source_fon
        df_sln['q_total_in'] = np.sum(q_in, axis=1) # Total mass of water in inflow (mm/d)
        df_sln['q_total_out'] = np.sum(q_out, axis=1) # Total mass of water in outflow (mm/d)

        dry_threshold = _dry_storage_threshold(self.params)
        c_din = pd.Series(
            np.divide(
                df_sln['m_din'],
                df_sln['s'],
                out=np.zeros(len(df_sln), dtype=float),
                where=df_sln['s'].to_numpy() > dry_threshold,
            ),
            index=df_sln.index,
        ) # DIN concentration in outflow (mg/L)
        c_don = pd.Series(
            np.divide(
                df_sln['m_don'],
                df_sln['s'],
                out=np.zeros(len(df_sln), dtype=float),
                where=df_sln['s'].to_numpy() > dry_threshold,
            ),
            index=df_sln.index,
        )  # DON concentration in outflow (mg/L)
        df_sln['c_din'] = c_din
        df_sln['c_don'] = c_don

        df_sln['m_din_total_flux_in'] = np.sum(q_in * c_din_in, axis=1) # Total mass of DIN in inflow (kg/km2/d)
        df_sln['m_don_total_flux_in'] = np.sum(q_in * c_don_in, axis=1) # Total mass of DON in inflow (kg/km2/d)
        df_sln['m_din_total_flux_out'] = df_sln['q_total_out'].values  * c_din # Total mass of DIN in outflow (kg/km2/d)
        df_sln['m_don_total_flux_out'] = df_sln['q_total_out'].values * c_don # Total mass of DON in outflow (kg/km2/d)

        return df_sln


NitrogenModel_SingleCV = NitrogenSoilLayer
