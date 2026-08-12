"""
GAC surrogate 모드 확인용 미니 flowsheet.
check_gac_structure.py와 달리 GAC 유닛모델을 ConcreteModel에 바로 붙이지 않고,
Feed -> GAC -> Product(처리수) / Product(흡착 제거량) 구조로 Arc 연결.
costing은 넣지 않음 (LCOW 계산 없이 breakthrough/성능 결과만 확인).

cphsdm_calaculation_method="surrogate"로 두면 a0, a1, b0~b4를 안 넣어도 됨
(Hand 1984 표 대신 PySMO RBF surrogate가 freund_ninv, N_Bi로부터
min_N_St/throughput을 직접 계산해줌). 나머지 값은 test_gac.py의
build_hand_surrogate()와 동일한 Hand, 1984 DCE 예제 값 그대로 사용.
"""

from pyomo.environ import ConcreteModel, TransformationFactory
from pyomo.network import Arc
import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Product

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.gac import GAC


def main():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MCASParameterBlock(
        solute_list=["DCE"],
        mw_data={"H2O": 0.018, "DCE": 0.09896},
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

    # --- 유입 조건 (Hand, 1984 DCE 예제) ---
    m.fs.feed.properties[0].pressure.fix(101325)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].fix(55555.55426666667)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "DCE"].fix(0.0002344381568310428)

    # --- 흡착 등온선 (Freundlich) ---
    m.fs.gac.freund_k.fix(37.9e-6 * (1e6**0.8316))
    m.fs.gac.freund_ninv.fix(0.8316)

    # --- GAC 입자 물성 ---
    m.fs.gac.particle_dens_app.fix(722)
    m.fs.gac.particle_dia.fix(0.00106)

    # --- 흡착조(bed) 사양 ---
    m.fs.gac.ebct.fix(300)  # seconds
    m.fs.gac.bed_voidage.fix(0.449)
    m.fs.gac.bed_length.fix(6)  # assumed

    # --- 설계 스펙 (파과 기준 농도비) ---
    m.fs.gac.conc_ratio_replace.fix(0.50)

    # --- 물질전달 파라미터 (film_transfer/surface_diffusion 둘 다 "fixed") ---
    m.fs.gac.kf.fix(3.29e-5)
    m.fs.gac.ds.fix(1.77e-13)
    # a0, a1, b0~b4는 surrogate 모드라 애초에 Var로 생성되지 않음 -> fix 대상 아님

    # --- 스케일링 ---
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e-4, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e4, index=("Liq", "DCE")
    )
    iscale.calculate_scaling_factors(m)

    # --- 자유도 확인 (0이어야 solve 가능) ---
    print("Degrees of freedom:", degrees_of_freedom(m))

    # --- 초기화: Feed -> propagate -> GAC -> propagate -> Product들 순서 ---
    m.fs.feed.initialize()
    propagate_state(m.fs.s01)
    m.fs.gac.initialize()
    propagate_state(m.fs.s02)
    propagate_state(m.fs.s03)
    m.fs.product.initialize()
    m.fs.adsorbed_removed.initialize()

    # --- solve ---
    solver = get_solver()
    results = solver.solve(m, tee=False)
    print("termination condition:", results.solver.termination_condition)

    # --- 결과 확인 ---
    m.fs.gac.report()

    return m, results


if __name__ == "__main__":
    main()
