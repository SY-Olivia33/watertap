"""
GAC (unit_models/gac.py, CPHSDM 물리 기반 모델) 구조 확인용 스크립트.
gac.py는 건드리지 않고, 이 파일에서 GAC를 import해서 pprint/solve로 확인한다.

GACZO(zero-order)와 달리 DB에서 값을 자동으로 불러오지 않고,
흡착등온선/베드/물질전달 파라미터를 전부 직접 fix해야 한다.
값은 test_gac.py의 build_hand() (Hand, 1984 DCE 제거 예제)를 그대로 사용.
"""

from pyomo.environ import ConcreteModel
import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.unit_models.gac import GAC


def main():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    # GAC는 WaterParameterBlock이 아니라 MCAS(multicomponent aqueous solution)
    # property package를 요구한다.
    m.fs.properties = MCASParameterBlock(
        solute_list=["DCE"],
        mw_data={"H2O": 0.018, "DCE": 0.09896},
        ignore_neutral_charge=True,
    )

    m.fs.unit = GAC(
        property_package=m.fs.properties,
        film_transfer_coefficient_type="fixed",
        surface_diffusion_coefficient_type="fixed",
    )

    # 구조(Var/Constraint 트리, bounds, fixed 여부)는 fix 하기 전에도 바로 볼 수 있음
    m.fs.unit.pprint()

    # --- 여기서부터 입력값 세팅 ---

    # 유입 조건
    unit_feed = m.fs.unit.process_flow.properties_in[0]
    unit_feed.pressure.fix(101325)
    unit_feed.temperature.fix(273.15 + 25)
    unit_feed.flow_mol_phase_comp["Liq", "H2O"].fix(55555.55426666667)
    unit_feed.flow_mol_phase_comp["Liq", "DCE"].fix(0.0002344381568310428)

    # 흡착 등온선 (Freundlich)
    m.fs.unit.freund_k.fix(37.9e-6 * (1e6**0.8316))
    m.fs.unit.freund_ninv.fix(0.8316)

    # GAC 입자 물성
    m.fs.unit.particle_dens_app.fix(722)
    m.fs.unit.particle_dia.fix(0.00106)

    # 흡착조(bed) 사양
    m.fs.unit.ebct.fix(300)  # seconds
    m.fs.unit.bed_voidage.fix(0.449)
    m.fs.unit.bed_length.fix(6)  # assumed

    # 설계 스펙 (파과 기준 농도비)
    m.fs.unit.conc_ratio_replace.fix(0.50)

    # 물질전달 파라미터
    m.fs.unit.kf.fix(3.29e-5)
    m.fs.unit.ds.fix(1.77e-13)
    m.fs.unit.a0.fix(3.68421)
    m.fs.unit.a1.fix(13.1579)
    m.fs.unit.b0.fix(0.784576)
    m.fs.unit.b1.fix(0.239663)
    m.fs.unit.b2.fix(0.484422)
    m.fs.unit.b3.fix(0.003206)
    m.fs.unit.b4.fix(0.134987)

    # 스케일링 (물리 기반 비선형 모델이라 GACZO와 달리 필수)
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e-4, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e4, index=("Liq", "DCE")
    )
    iscale.calculate_scaling_factors(m)

    # 자유도 확인 (0이어야 solve 가능)
    print("Degrees of freedom:", degrees_of_freedom(m))

    # 초기화 (비선형성이 커서 initialize 없이 바로 solve하면 수렴 실패 가능)
    m.fs.unit.initialize()

    # solve
    solver = get_solver()
    results = solver.solve(m, tee=False)
    print(results.solver.termination_condition)

    # 결과 확인
    m.fs.unit.report()


if __name__ == "__main__":
    main()
