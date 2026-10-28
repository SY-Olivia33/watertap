"""
GAC gravity-type contactor 최적화 flowsheet.
baseline(고정값) 버전부터 완성 후, EBCT, loading rate 등을
unfix해서 최적화로 확장
contactor dimension과 contactor number도 최적화 대상
+ on-site regen, off-site regen(non haz vs. haz), disposal(non haz vs. haz)의 CAPEX와 OPEX를 costing에 반영
"""

import pyomo.environ as pyo
from pyomo.network import Arc
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Product
import idaes.core.util.scaling as iscale
import math

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.gac import GAC, CPHSDMCalculationMethod
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import ContactorType


def main():

    m = build(
        material_flow_basis="molar",  ## PFAS species -> material_flow_basis="mass"로 넘기는 게 나을지도
        film_transfer_coefficient_type="calculated",
        surface_diffusion_coefficient_type="calculated",
        diffusivity_calculation="HaydukLaudie",
        cost_contactor_type="gravity",
        solute = "solute"
    )
    ## 일단 지금은 cphsdm input으로 돌려보고 나중에 surrogate으로 바꿀 거라서 이후 cphsdm_method="surrogate" 넘겨주기
    initialize(m)
    res = optimize(m)
    print_results(m, res) ## function for printing results should be added
                        ## print_results should include res.solver.termination_condition

    return m

## contactor type has pressure type and gravity type
## For now, only gravity type considered - lines for pressure type will be added
## Roadmap : optimization for -
##      EBCT, loading rate(calculated by bed length & EBCT), contactor number, contactor dimension, and resulting optimal LCOW
##      consider range limit for each var
##      for costing, regen(on-site, off-site(non-haz & haz), disposal(non-haz & haz)
##      unfixed vars can be fixed by user

def build(
    material_flow_basis="molar",
    film_transfer_coefficient_type="fixed",
    surface_diffusion_coefficient_type="fixed",
    diffusivity_calculation="none",
    cost_contactor_type="pressure",
    solute = "solute",
    cphsdm_method = "input"
):
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    cost_contactor_type=ContactorType(cost_contactor_type)
        ## "gravity" -> ContactorType.gravity
    solute_data = {"solute":dict(mw = 0.08, mv = 1e-4, freund_k = 10, freund_ninv = 0.9, diffus = 1e-9, conc_feed = 0.1),}
    data = solute_data[solute]
        ## conc_feed [kg/m3]
        ## other PFAS data will be added
        ## "solute" reference from flowsheets\gac\gac.py ; 잘 돌아가나 확인용
    if (
        film_transfer_coefficient_type == "calculated"
        or surface_diffusion_coefficient_type == "calculated"  ## kf or ds is unfixed -> 어떤 property package를 넘겨줄 것인가
    ):
        if diffusivity_calculation == "none":  ## use the diffusivity data from solute_data, so data[diffus] needed
            m.fs.properties = MCASParameterBlock(
                material_flow_basis = material_flow_basis,
                ignore_neutral_charge = True,
                solute_list = [solute],
                mw_data = {"H2O": 0.018, solute: data["mw"]},
                diffus_calculation = diffusivity_calculation,
                diffusivity_data = {("Liq", solute): data["diffus"]},
            )
        else:  ## calculate diffusivity data using molar volume, with HayduckLaudie, so data["mv"] needed
            ## HayduckLaudie is for nonelectrolyte molecules. PFAS는 음이온으로 존재 -> 문헌값 사용하는 것이 좋음
            m.fs.properties = MCASParameterBlock(
                material_flow_basis = material_flow_basis,
                ignore_neutral_charge = True,
                solute_list = [solute],
                mw_data = {"H2O": 0.018, solute: data["mw"]},
                diffus_calculation = diffusivity_calculation,
                molar_volume_data = {("Liq", solute): data["mv"]},
            )
    else:  ## diffusivity coefficient unnecessary ; kf and ds are fixed
        m.fs.properties = MCASParameterBlock(
            material_flow_basis = material_flow_basis,
            ignore_neutral_charge = True,
            solute_list = [solute],
            mw_data = {"H2O": 0.018, solute: data["mw"]},
        )
       ## defining property package ; if/else statements are for considering diffusivity coefficient (to include or not)

    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.gac = GAC(
        property_package=m.fs.properties,
        film_transfer_coefficient_type=film_transfer_coefficient_type,
        surface_diffusion_coefficient_type=surface_diffusion_coefficient_type,
        cphsdm_calaculation_method=CPHSDMCalculationMethod[cphsdm_method],
    )
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.adsorbed_removed = Product(property_package=m.fs.properties)

    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.gac.inlet)
    m.fs.s02 = Arc(source=m.fs.gac.outlet, destination=m.fs.product.inlet)
    m.fs.s03 = Arc(source=m.fs.gac.adsorbed, destination=m.fs.adsorbed_removed.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    add_costing(m, cost_contactor_type=cost_contactor_type)

    ## scaling
    flow_vol_q = 0.043813  ## [m3/s], reference 값
    flow_mass_water = flow_vol_q * 1000  ## [kg/s]
    flow_mass_solute = flow_vol_q * data["conc_feed"]  ## [kg/s]

    if material_flow_basis == "molar":
        flow_water = flow_mass_water / 0.018  ## [mol/s]
        flow_solute = flow_mass_solute / data["mw"]  ## [mol/s]
        flow_var = "flow_mol_phase_comp"
    else:
        flow_water = flow_mass_water
        flow_solute = flow_mass_solute
        flow_var = "flow_mass_phase_comp"

    water_sf = 1/flow_water
    solute_sf = 1/flow_solute
    m.fs.properties.set_default_scaling(flow_var, water_sf, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(flow_var, solute_sf, index=("Liq", solute))

    m.fs.feed.properties[0].conc_mass_phase_comp   ## touch
    m.fs.feed.properties[0].flow_vol_phase["Liq"]  ## feed의 conc와 flow_vol도 scaling 대상에 넣기 위함

    iscale.calculate_scaling_factors(m)

    ## feed
    m.fs.feed.properties[0].temperature.fix(273.15 + 25) ## feed, [K]
    m.fs.feed.properties[0].pressure.fix(101325) ## feed,  [Pa]
    feed_flow = getattr(m.fs.feed.properties[0], flow_var)  ## = m.fs.feed.properties[0].flow_mol_phase_comp or flow_mass_phase_comp
    feed_flow["Liq", "H2O"].fix(flow_water)
    feed_flow["Liq", solute].fix(flow_solute)

    ## gac ; performance specifications ->
    m.fs.gac.freund_k.fix(data["freund_k"])
    m.fs.gac.freund_ninv.fix(data["freund_ninv"])
    m.fs.gac.kf.fix(5e-5)  ## fixed 옵션일 때만
    m.fs.gac.ds.fix(2e-13)  ## fixed 옵션일 때만

    ## gac ; particle specifications
    m.fs.gac.particle_dens_app.fix(750)
    m.fs.gac.particle_dia.fix(0.001)

    ## bed & design spec
    m.fs.gac.bed_voidage.fix(0.4)
    m.fs.gac.ebct.fix(600)  ## optimize에서 unfix
    m.fs.gac.bed_length.fix(6)  ## optimize에서 unfix
    ## velocity_sup = loading rate = bed length / ebct
    m.fs.gac.conc_ratio_replace.fix(0.50)

    ## parameters
    if cphsdm_method == "input":   ## reference : 1/n = 0.9 기준. alternate if 1/n changes as solute changes
        m.fs.gac.a0.fix(3.68421)
        m.fs.gac.a1.fix(13.1579)
        m.fs.gac.b0.fix(0.784576)
        m.fs.gac.b1.fix(0.239663)
        m.fs.gac.b2.fix(0.484422)
        m.fs.gac.b3.fix(0.003206)
        m.fs.gac.b4.fix(0.134987)

    if film_transfer_coefficient_type == "calculated":
        m.fs.gac.kf.unfix()
        m.fs.gac.shape_correction_factor.fix()
    if surface_diffusion_coefficient_type == "calculated":
        m.fs.gac.ds.unfix()
        m.fs.gac.particle_porosity.fix()
        m.fs.gac.tort.fix()
        m.fs.gac.spdfr.fix()

    return m


def initialize():

def optimize():

def add_costing(m, cost_contactor_type = "pressure"):
    m.fs.costing = WaterTAPCosting
    m.fs.costing.base_currency = pyo.units.USD_2021
    m.fs.gac.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"cost_contactor_type": cost_contactor_type},
    )

def print_results():



if __name__ == "__main__":
    m = main()