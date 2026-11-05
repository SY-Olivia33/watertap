"""
GAC gravity-type CAPEX sweep (1 ~ 100 MGD), WaterTAP cost_gac_gravity 사용

CAPEX = TIC(2.0) x (contactor_cost + adsorbent_cost + other_process_cost)
  - contactor_cost      : 콘크리트 basin (운전 + 예비 contactor)
  - adsorbent_cost      : GAC 초기 충전
  - other_process_cost  : 배관, 계장, 제어 등 기타

실행 (watertap 폴더에서):
    python Study/Project/gac_capex_sweep.py            -> WaterTAP 기본값 (contactor 1+1)
    python Study/Project/gac_capex_sweep.py --scaled   -> 유량에 따라 contactor 수 증가

결과: 화면에 표 출력 + 같은 폴더에 CSV 저장
"""

import argparse
import csv
import math
import os

import pyomo.environ as pyo
from pyomo.network import Arc
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Product
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.gac import GAC
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import ContactorType

## ---------------- 설정값 (여기만 바꾸면 됨) ----------------
FLOWS_MGD = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EBCT = 600  ## [s], WaterTAP 기본값
MGD_TO_M3S = 0.0438126364  ## 1 MGD = 0.0438 m3/s
FT = 0.3048

## WaterTAP 기본값 모드
DEFAULT_BED_LENGTH = 6.0  ## [m]
## --scaled 모드 : contactor 하나 최대 30ft x 30ft x 10ft
SCALED_BED_LENGTH = 10 * FT  ## [m]
UNIT_VOLUME_MAX = (30 * FT) ** 2 * (10 * FT)  ## [m3], 약 255 m3
## -----------------------------------------------------------


def contactor_numbers(q_mgd, scaled):
    if not scaled:
        return 1, 1  ## WaterTAP 기본값
    bed_volume = q_mgd * MGD_TO_M3S * EBCT
    n_op = max(1, math.ceil(bed_volume / UNIT_VOLUME_MAX))
    n_red = 2 if q_mgd >= 10 else 1
    return n_op, n_red


def build_and_solve(q_mgd, scaled):
    q = q_mgd * MGD_TO_M3S  ## [m3/s]
    bed_length = SCALED_BED_LENGTH if scaled else DEFAULT_BED_LENGTH
    n_op, n_red = contactor_numbers(q_mgd, scaled)

    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        material_flow_basis="molar",
        ignore_neutral_charge=True,
        solute_list=["solute"],
        mw_data={"H2O": 0.018, "solute": 0.08},
    )
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.gac = GAC(
        property_package=m.fs.properties,
        film_transfer_coefficient_type="fixed",
        surface_diffusion_coefficient_type="fixed",
    )
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.adsorbed_removed = Product(property_package=m.fs.properties)
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.gac.inlet)
    m.fs.s02 = Arc(source=m.fs.gac.outlet, destination=m.fs.product.inlet)
    m.fs.s03 = Arc(source=m.fs.gac.adsorbed, destination=m.fs.adsorbed_removed.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    ## feed (용질은 reference flowsheet의 placeholder; CAPEX에는 영향 없음)
    flow_water = q * 1000 / 0.018  ## [mol/s]
    flow_solute = q * 0.1 / 0.08  ## [mol/s]
    m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1 / flow_water, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mol_phase_comp", 1 / flow_solute, index=("Liq", "solute"))
    m.fs.feed.properties[0].temperature.fix(298.15)
    m.fs.feed.properties[0].pressure.fix(101325)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].fix(flow_water)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "solute"].fix(flow_solute)

    ## gac (reference flowsheet 기본값)
    gac = m.fs.gac
    gac.freund_k.fix(10)
    gac.freund_ninv.fix(0.9)
    gac.kf.fix(5e-5)
    gac.ds.fix(2e-13)
    gac.particle_dens_app.fix(750)
    gac.particle_dia.fix(0.001)
    gac.bed_voidage.fix(0.4)
    gac.ebct.fix(EBCT)
    gac.bed_length.fix(bed_length)
    gac.conc_ratio_replace.fix(0.50)
    gac.a0.fix(3.68421)
    gac.a1.fix(13.1579)
    gac.b0.fix(0.784576)
    gac.b1.fix(0.239663)
    gac.b2.fix(0.484422)
    gac.b3.fix(0.003206)
    gac.b4.fix(0.134987)

    ## costing (gravity)
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2021
    gac.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"contactor_type": ContactorType.gravity},
    )
    m.fs.costing.cost_process()
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.gac_gravity.regen_frac.fix(0.7)
    m.fs.costing.gac_gravity.num_contactors_op.fix(n_op)
    m.fs.costing.gac_gravity.num_contactors_redundant.fix(n_red)

    iscale.calculate_scaling_factors(m)

    ## initialize & solve
    m.fs.feed.initialize(outlvl=idaeslog.CRITICAL)
    propagate_state(m.fs.s01)
    gac.initialize(outlvl=idaeslog.CRITICAL)
    propagate_state(m.fs.s02)
    propagate_state(m.fs.s03)
    m.fs.product.initialize(outlvl=idaeslog.CRITICAL)
    m.fs.adsorbed_removed.initialize(outlvl=idaeslog.CRITICAL)
    gac.costing.initialize()
    m.fs.costing.initialize()
    res = get_solver().solve(m)
    pyo.assert_optimal_termination(res)

    return {
        "Q_MGD": q_mgd,
        "n_op": n_op,
        "n_red": n_red,
        "contactor_cost": pyo.value(gac.costing.contactor_cost),
        "adsorbent_cost": pyo.value(gac.costing.adsorbent_cost),
        "other_process_cost": pyo.value(gac.costing.other_process_cost),
        "total_CAPEX": pyo.value(gac.costing.capital_cost),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled", action="store_true", help="유량에 따라 contactor 수 증가")
    args = parser.parse_args()

    mode = "scaled" if args.scaled else "default"
    print(f"--- GAC gravity CAPEX [USD 2021], mode = {mode} ---")
    print(f"{'Q[MGD]':>6} {'n_op+red':>8} {'contactor_cost':>15} {'adsorbent_cost':>15} {'other_process':>15} {'Total CAPEX':>15}")

    rows = []
    for q in FLOWS_MGD:
        try:
            r = build_and_solve(q, args.scaled)
        except (RuntimeError, ValueError) as e:
            print(f"{q:>6} solve failed ({type(e).__name__})")
            continue
        rows.append(r)
        print(f"{q:>6} {str(r['n_op']) + '+' + str(r['n_red']):>8} "
              f"{r['contactor_cost']:>15,.0f} {r['adsorbent_cost']:>15,.0f} "
              f"{r['other_process_cost']:>15,.0f} {r['total_CAPEX']:>15,.0f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gac_capex_{mode}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
