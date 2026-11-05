"""
GAC surrogate 모드 확인용 미니 flowsheet + gravity costing.
check_gac_structure.py와 달리 GAC 유닛모델을 ConcreteModel에 바로 붙이지 않고,
Feed -> GAC -> Product(처리수) / Product(흡착 제거량) 구조로 Arc 연결.

cphsdm_calaculation_method="surrogate"로 두면 a0, a1, b0~b4를 안 넣어도 됨
(Hand 1984 표 대신 PySMO RBF surrogate가 freund_ninv, N_Bi로부터
min_N_St/throughput을 직접 계산해줌).

입력값은 전부 WaterTAP reference flowsheet(watertap/flowsheets/gac/gac.py) 기본값.
(이전 버전은 test_gac.py의 Hand, 1984 DCE 예제 값을 썼음)
유량만 MGD로 바꿔가며 gravity CAPEX (contactor / adsorbent / other / total) 확인.

실행 (watertap 폴더에서):
    python Study/Practice/gac_surrogate_flowsheet.py
"""

from pyomo.environ import ConcreteModel, TransformationFactory, value, units as pyunits
from pyomo.network import Arc
import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Product

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.gac import GAC
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import ContactorType

import math

FLOWS_MGD = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
MGD_TO_M3S = 0.0438126364  # 1 MGD = 0.0438 m3/s (reference flow 0.043813 m3/s = 1 MGD)
EBCT = 600  # [s], WaterTAP 기본값

# contactor 개수 반복 범위
N_OP_RANGE = range(1, 21)
# contactor 1개 부피 상한 [m3]
# gravity contactor 비용식 75131 + 735.55*V - 1.01827*V^2 이 V = 735.55/(2*1.01827) = 361 m3에서 최대
# -> 이보다 크면 contactor가 클수록 싸지는 비현실적인 구간이라 제외
UNIT_VOLUME_MAX = 735.55 / (2 * 1.01827)


def num_redundant(flow_mgd):
    # 1 MGD(10 MGD 미만)는 예비 1개, 10 MGD부터는 예비 2개
    return 1 if flow_mgd < 10 else 2


def main(flow_mgd=1.0, n_op=1, n_red=1, verbose=True):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # --- 물성 (reference 기본 용질 "solute") ---
    m.fs.properties = MCASParameterBlock(
        solute_list=["solute"],
        mw_data={"H2O": 0.018, "solute": 0.08},
        ignore_neutral_charge=True,
    )

    # --- 블록 ---
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.gac = GAC(
        property_package=m.fs.properties,
        film_transfer_coefficient_type="fixed",
        surface_diffusion_coefficient_type="fixed",
        cphsdm_calaculation_method="surrogate",  # a0,a1,b0~b4 불필요
    )
    m.fs.product = Product(property_package=m.fs.properties)  # 처리수(outlet)
    m.fs.adsorbed_removed = Product(property_package=m.fs.properties)  # 흡착 제거량(adsorbed)

    # --- 스트림 연결 ---
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.gac.inlet)
    m.fs.s02 = Arc(source=m.fs.gac.outlet, destination=m.fs.product.inlet)
    m.fs.s03 = Arc(source=m.fs.gac.adsorbed, destination=m.fs.adsorbed_removed.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # --- 유입 조건 (reference: 1 MGD일 때 H2O 2433.81 mol/s, solute 0.0548 mol/s = 0.1 kg/m3) ---
    flow_vol = flow_mgd * MGD_TO_M3S  # [m3/s]
    flow_water = flow_vol * 1000 / 0.018  # [mol/s]
    flow_solute = flow_vol * 0.1 / 0.08  # [mol/s]
    m.fs.feed.properties[0].pressure.fix(101325)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].fix(flow_water)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "solute"].fix(flow_solute)

    # --- 흡착 등온선 (Freundlich) ---
    m.fs.gac.freund_k.fix(10)
    m.fs.gac.freund_ninv.fix(0.9)

    # --- GAC 입자 물성 ---
    m.fs.gac.particle_dens_app.fix(750)
    m.fs.gac.particle_dia.fix(0.001)

    # --- 흡착조(bed) 사양 ---
    m.fs.gac.ebct.fix(EBCT)  # seconds
    m.fs.gac.bed_voidage.fix(0.4)
    m.fs.gac.bed_length.fix(6)  # m

    # --- 설계 스펙 (파과 기준 농도비) ---
    m.fs.gac.conc_ratio_replace.fix(0.50)

    # --- 물질전달 파라미터 (film_transfer/surface_diffusion 둘 다 "fixed") ---
    m.fs.gac.kf.fix(5e-5)
    m.fs.gac.ds.fix(2e-13)
    # a0, a1, b0~b4는 surrogate 모드라 애초에 Var로 생성되지 않음 -> fix 대상 아님

    # --- costing (gravity, reference 기본값) ---
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyunits.USD_2021
    m.fs.gac.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"contactor_type": ContactorType.gravity},
    )
    m.fs.costing.cost_process()
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.gac_gravity.regen_frac.fix(0.7)
    m.fs.costing.gac_gravity.num_contactors_op.fix(n_op)
    m.fs.costing.gac_gravity.num_contactors_redundant.fix(n_red)

    # --- 스케일링 (유량에 맞춰서) ---
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1 / flow_water, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1 / flow_solute, index=("Liq", "solute")
    )
    iscale.calculate_scaling_factors(m)

    # --- 자유도 확인 (0이어야 solve 가능) ---
    if verbose:
        print("Degrees of freedom:", degrees_of_freedom(m))

    # --- 초기화: Feed -> propagate -> GAC -> propagate -> Product들 -> costing 순서 ---
    m.fs.feed.initialize()
    propagate_state(m.fs.s01)
    m.fs.gac.initialize()
    propagate_state(m.fs.s02)
    propagate_state(m.fs.s03)
    m.fs.product.initialize()
    m.fs.adsorbed_removed.initialize()
    m.fs.gac.costing.initialize()
    m.fs.costing.initialize()

    # --- solve ---
    solver = get_solver()
    results = solver.solve(m, tee=False)
    if verbose:
        print("termination condition:", results.solver.termination_condition)

    return m, results


def get_capex(m):
    c = m.fs.gac.costing
    return {
        "contactor_cost": value(c.contactor_cost),
        "adsorbent_cost": value(c.adsorbent_cost),
        "other_process_cost": value(c.other_process_cost),
        "total_capex": value(c.capital_cost),
    }


def find_cheapest(flow_mgd):
    """contactor 개수 1~20개 중 CAPEX가 최소인 경우 찾기 (contactor 1개 부피 <= UNIT_VOLUME_MAX)"""
    n_red = num_redundant(flow_mgd)
    bed_volume = flow_mgd * MGD_TO_M3S * EBCT  # [m3], 운전 contactor 전체 부피
    candidates = [n for n in N_OP_RANGE if bed_volume / n <= UNIT_VOLUME_MAX]
    if not candidates:
        return None

    # 가장 적은 개수로 모델을 한 번 만들고, 이후엔 개수만 바꿔서 다시 solve (매번 새로 만들면 느림)
    m, results = main(flow_mgd=flow_mgd, n_op=candidates[0], n_red=n_red, verbose=False)
    solver = get_solver()
    best = None
    for n in candidates:
        m.fs.costing.gac_gravity.num_contactors_op.fix(n)
        try:
            results = solver.solve(m, tee=False)
        except ValueError:
            continue
        if str(results.solver.termination_condition) != "optimal":
            continue
        cost = get_capex(m)
        if best is None or cost["total_capex"] < best["total_capex"]:
            best = dict(flow_mgd=flow_mgd, n_op=n, n_red=n_red,
                        unit_volume=bed_volume / n, **cost)
    return best


def print_result(r):
    print(f"=== Flow: {r['flow_mgd']} MGD ===")
    print(f"  Number of contactors (operating + redundant): {r['n_op']} + {r['n_red']} = {r['n_op'] + r['n_red']}")
    print(f"  Unit contactor volume: {r['unit_volume']:,.2f} [m3]")
    print(f"  Contactor cost:     {r['contactor_cost']:>16,.2f} [USD_2021]")
    print(f"  Adsorbent cost:     {r['adsorbent_cost']:>16,.2f} [USD_2021]")
    print(f"  Other process cost: {r['other_process_cost']:>16,.2f} [USD_2021]")
    print(f"  Total CAPEX:        {r['total_capex']:>16,.2f} [USD_2021]  (= TIC 2.0 x sum of the three)")


if __name__ == "__main__":  # Windows에서 import 시 반복문이 한 번 더 도는 것 방지
    print("GAC gravity CAPEX - cheapest number of contactors (1~20), unit volume <= "
          f"{UNIT_VOLUME_MAX:.2f} m3, EBCT {EBCT} s")
    for q in FLOWS_MGD:
        r = find_cheapest(q)
        if r is None:
            print(f"=== Flow: {q} MGD ===")
            print(f"  no feasible contactor number in {N_OP_RANGE}")
            continue
        print_result(r)
