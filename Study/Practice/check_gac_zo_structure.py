"""
GACZO 모델 구조 확인용 스크립트.
gac_zo.py는 건드리지 않고, 이 파일에서 GACZO를 import해서 pprint/solve로 확인한다.
"""

from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver
from watertap.unit_models.zero_order import GACZO
from watertap.core.wt_database import Database
from watertap.core.zero_order_properties import WaterParameterBlock


def main():
    m = ConcreteModel()
    m.db = Database()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.params = WaterParameterBlock(solute_list=["tss", "nonvolatile_toc"])

    m.fs.unit = GACZO(property_package=m.fs.params, database=m.db)
    m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(10000)
    m.fs.unit.inlet.flow_mass_comp[0, "tss"].fix(1)
    m.fs.unit.inlet.flow_mass_comp[0, "nonvolatile_toc"].fix(1)

    # solve 안 해도 구조(Var/Constraint 트리, bounds, fixed 여부)는 바로 볼 수 있음
    m.fs.unit.pprint()

    # 콘솔에 너무 많이 나오면 파일로 저장
    # with open("gac_structure.txt", "w") as f:
    #     m.fs.unit.pprint(ostream=f)

    # --- 여기서부터 solve ---

    # 1) DB에서 GAC 전용 파라미터 로드 (empty_bed_contact_time, removal_frac 등 자동 fix)
    m.fs.unit.load_parameters_from_database()

    # 2) 자유도 확인 (0이어야 solve 가능)
    print("Degrees of freedom:", degrees_of_freedom(m.fs.unit))

    # 3) solve
    solver = get_solver()
    results = solver.solve(m.fs.unit, tee=False)
    print(results.solver.termination_condition)

    # 4) 결과 확인 - 구조가 아니라 실제 값이 궁금할 땐 report()가 더 보기 편함
    m.fs.unit.report()


if __name__ == "__main__":
    main()
