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
from idaes.models.unit_models import Feed, Product
import idaes.core.util.scaling as iscale
import math

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.gac import GAC, CPHSDMCalculationMethod
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import ContactorType
from pyomo.util.check_units import assert_units_consistent
from watertap.core.util.initialization import assert_degrees_of_freedom


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
    cost_params = m.fs.costing.gac_gravity
    optimal_j = None
    optimal_lcow = None
    for j in range (1, 21):
        cost_params.num_contactors_op.fix(j)
        try:
            optimize(m)
        except (RuntimeError, ValueError):
            continue
        lcow = pyo.value(m.fs.costing.LCOW)
        if optimal_lcow is None or lcow < optimal_lcow:
            optimal_lcow = lcow
            optimal_j = j
    if optimal_j is None:
        raise RuntimeError("Convergence failed")
    cost_params.num_contactors_op.fix(optimal_j)
    res = optimize(m)
    print_results(m, res, cost_contactor_type="gravity") ## function for printing results should be added
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
        else:  ## calculate diffusivity data using molar volume, with HaydukLaudie, so data["mv"] needed
            ## HaydukLaudie is for nonelectrolyte molecules. PFAS는 음이온으로 존재 -> 문헌값 사용하는 것이 좋음
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

    flow_vol_q = 0.043813  ## [m3/s], reference 값
    flow_mass_water = flow_vol_q * 1000  ## [kg/s]
    flow_mass_solute = flow_vol_q * data["conc_feed"]  ## [kg/s]

    add_costing(m, cost_contactor_type=cost_contactor_type, flow_vol_q=flow_vol_q)

    ## scaling

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
    m.fs.gac.bed_length.fix(3)  ## optimize에서 unfix
    ## velocity_sup = loading rate = bed length / ebct
    m.fs.gac.conc_ratio_replace.fix(0.50)

    ## new variables
    m.fs.unit_width = pyo.Var(initialize=3, units=pyo.units.m)
    m.fs.unit_area = pyo.Var(initialize=9, units=pyo.units.m ** 2)
    m.fs.eq_unit_area = pyo.Constraint(expr=m.fs.unit_area == m.fs.unit_width ** 2)
    m.fs.eq_bed_area = pyo.Constraint(expr=m.fs.gac.bed_area == m.fs.costing.gac_gravity.num_contactors_op * m.fs.unit_area)

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

    ## 목적함수
    m.fs.obj = pyo.Objective(expr=m.fs.costing.LCOW)

    return m


def initialize(m, solver=None):
    if solver is None:
        solver = get_solver()

    ## check model ; units, DOF
    assert_units_consistent(m)
    assert_degrees_of_freedom(m, 0)

    ## initialize
    m.fs.feed.initialize()
    propagate_state(m.fs.s01)
    m.fs.gac.initialize()
    propagate_state(m.fs.s02)
    propagate_state(m.fs.s03)
    m.fs.product.initialize()
    m.fs.adsorbed_removed.initialize()

    ## costing initialize
    m.fs.gac.costing.initialize()
    m.fs.costing.initialize()
    ## gac.costing을 먼저 initialize 해줘야 costing이 gac.costing의 값을 받아서 제대로 된 초기값 계산 가능

    ## solve
    res = solver.solve(m)
    pyo.assert_optimal_termination(res)

    return res


def optimize(m, solver=None):
    ## reference ; EBCT(600s) < minimum EBCT(2867s) (to satisfy CPHSDM's constant pattern assumption)
    ## Therefore, constraints that makes optimized EBCT larger than the minimum EBCT should be added (ebct >= min_ebct)

    if solver is None:
        solver = get_solver()

    m.fs.unit_width.setlb(6 * pyo.units.ft)
    m.fs.unit_width.setub(30 * pyo.units.ft)
    m.fs.gac.bed_length.unfix()
    m.fs.gac.bed_length.setlb(3 * pyo.units.ft)
    m.fs.gac.bed_length.setub(10 * pyo.units.ft)
    m.fs.gac.ebct.unfix()
    m.fs.gac.ebct.setlb(456)
    m.fs.gac.ebct.setub(1560)

    ## check model ; units
    assert_units_consistent(m)
    ## solve
    res = solver.solve(m)
    pyo.assert_optimal_termination(res)

    return res


def add_costing(m, cost_contactor_type = "pressure", flow_vol_q = 0.043813):
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2021  ## 년도 어떻게 할지??
    m.fs.gac.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"contactor_type": cost_contactor_type},
        ## costing_method=어쩌구... regen 추가할 때 cost_gac를 대체할 새 함수를 넣기
    )
    ## add flowsheet level blocks
    m.fs.costing.cost_process()  ## cost_process method 호출
    treated_flow = m.fs.product.properties[0].flow_vol  ## product의 flow_vol
    m.fs.costing.add_LCOW(treated_flow)  ## treated_flow 이용해 LCOW 계산

    flow_vol_q_mgd = pyo.units.convert_value(flow_vol_q, from_units=pyo.units.m ** 3 / pyo.units.s, to_units=pyo.units.Mgallon / pyo.units.day, )

    ## costing specifications
    if cost_contactor_type == ContactorType.pressure:
        m.fs.costing.gac_pressure.regen_frac.fix(0.7)
        m.fs.costing.gac_pressure.num_contactors_op.fix(1)
        m.fs.costing.gac_pressure.num_contactors_redundant.fix(1)
    else:  ## gravity
        m.fs.costing.gac_gravity.regen_frac.fix(0.7)
        m.fs.costing.gac_gravity.num_contactors_op.fix(1)
        if flow_vol_q_mgd >= 10:
            m.fs.costing.gac_gravity.num_contactors_redundant.fix(2)
        else:  ## Q<1 일 때는 EPA 기준 gravity type에서 주어진 redundant contactor number 없음
            m.fs.costing.gac_gravity.num_contactors_redundant.fix(1)


def print_results(m, res, cost_contactor_type):

    print(f"termination: {res.solver.termination_condition}")

    print("--- Optimization Results ---")
    print(f"EBCT: {pyo.value(m.fs.gac.ebct):.2f} [s]")
    print(f"Bed Length: {pyo.value(m.fs.gac.bed_length):.2f} [m]")
    print(f"Loading Rate: {pyo.value(m.fs.gac.velocity_sup)*3600:.2f} [m/h]")

    contactor_numbers = getattr(m.fs.costing, f"gac_{cost_contactor_type}")
    n_op = int(pyo.value(contactor_numbers.num_contactors_op))
    n_red = int(pyo.value(contactor_numbers.num_contactors_redundant))

    print(f"Total bed area / Total bed volume: {pyo.value(m.fs.gac.bed_area):.2f} [m2] / {pyo.value(m.fs.gac.bed_volume):.2f} [m3]")
    ## operating만 고려
    print(f"Unit bed width / Unit bed area / Unit bed volume: {pyo.value(m.fs.unit_width):.2f} [m] / {pyo.value(m.fs.unit_area):.2f} [m2] / {pyo.value(m.fs.gac.bed_volume)/n_op:.2f} [m3]")
    ## ft로 할지 아니면 그냥 m로 해도 ㄱㅊ을지?? 엑셀이랑 비교할 거 생각하면 ft로 변환하는 게 낫나 싶긴 한데 음
    print(f"Number of total contactors: {n_op + n_red}")
    print(f"Number of operating contactors: {n_op}")
    print(f"Number of redundant contactors: {n_red}")
    print(f"GAC Mass filled in bed: {pyo.value(m.fs.gac.bed_mass_gac):.2f} [kg]")
    print(f"EBCT Minimal: {pyo.value(m.fs.gac.min_ebct):.2f} [s]")
    ## regen type도 어떤 걸로 했는지 나중에 추가해줘야 함

    print("--- Performance Results ---")
    print(f"Bed Volumes Treated: {pyo.value(m.fs.gac.bed_volumes_treated):.0f} [BV]")  ## = bed life expressed in BV
    print(f"Bed Life: {pyo.value(m.fs.gac.operational_time)/86400:.1f} [days]")
    print(f"Mass Removed: {pyo.value(m.fs.gac.mass_adsorbed)} [kg]")
    print(f"GAC Usage Rate: {pyo.value(m.fs.gac.gac_usage_rate)*86400:.1f} [kg/day]")

    print("--- Cost ---")
    print(f"CAPEX for GAC: {pyo.value(m.fs.gac.costing.capital_cost):,.2f} [$]")
    print(f"OPEX for GAC: {pyo.value(m.fs.gac.costing.fixed_operating_cost):,.2f} [$/year]")
    print(f"Total CAPEX: {pyo.value(m.fs.costing.total_capital_cost):,.2f} [$]")
    print(f"Total OPEX: {pyo.value(m.fs.costing.total_operating_cost):,.2f} [$/year]")
    print(f"Total Annualized Cost: {pyo.value(m.fs.costing.total_annualized_cost):,.2f} [$/year]")
    print(f"LCOW: {pyo.value(m.fs.costing.LCOW):,.4f} [$/m3]")


if __name__ == "__main__":
    m = main()