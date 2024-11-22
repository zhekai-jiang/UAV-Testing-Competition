import random
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
import math
from pymoo.core.variable import Real

class MHSGenerator(object):
    min_size = Obstacle.Size(2, 2, 10)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    def __init__(self, case_study_file: str) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []

        class ObstacleProblem(Problem):
            def __init__(self):
                vars = {}
                vars["l"] = Real(bounds=(MHSGenerator.min_size.l, MHSGenerator.max_size.l))
                vars["w"] = Real(bounds=(MHSGenerator.min_size.w, MHSGenerator.max_size.w))
                vars["h"] = Real(bounds=(MHSGenerator.min_size.h, MHSGenerator.max_size.h))
                vars["x"] = Real(bounds=(MHSGenerator.min_position.x, MHSGenerator.max_position.x))
                vars["y"] = Real(bounds=(MHSGenerator.min_position.y, MHSGenerator.max_position.y))
                vars["r"] = Real(bounds=(MHSGenerator.min_position.r, MHSGenerator.max_position.r))
                super().__init__(vars=vars, n_obj=1)

            def _evaluate(self, x, out, *args, **kwargs):
                # print(x)
                l, w, h, x_pos, y_pos, r = x[0]["l"], x[0]["w"], x[0]["h"], x[0]["x"], x[0]["y"], x[0]["r"]

                # Calculate the rotated bounds for all four corners of the rectangle
                half_l = l / 2
                half_w = w / 2
                corners = [
                    (x_pos + half_l * math.cos(math.radians(r)) - half_w * math.sin(math.radians(r)),
                     y_pos + half_l * math.sin(math.radians(r)) + half_w * math.cos(math.radians(r))),
                    (x_pos - half_l * math.cos(math.radians(r)) - half_w * math.sin(math.radians(r)),
                     y_pos - half_l * math.sin(math.radians(r)) + half_w * math.cos(math.radians(r))),
                    (x_pos + half_l * math.cos(math.radians(r)) + half_w * math.sin(math.radians(r)),
                     y_pos + half_l * math.sin(math.radians(r)) - half_w * math.cos(math.radians(r))),
                    (x_pos - half_l * math.cos(math.radians(r)) + half_w * math.sin(math.radians(r)),
                     y_pos - half_l * math.sin(math.radians(r)) - half_w * math.cos(math.radians(r)))
                ]

                # Check if all corners are within the allowed bounds after rotation
                valid = all(
                    MHSGenerator.min_position.x <= x <= MHSGenerator.max_position.x and
                    MHSGenerator.min_position.y <= y <= MHSGenerator.max_position.y
                    for x, y in corners
                )

                if (MHSGenerator.min_size.l <= l <= MHSGenerator.max_size.l and
                    MHSGenerator.min_size.w <= w <= MHSGenerator.max_size.w and
                    MHSGenerator.min_size.h <= h <= MHSGenerator.max_size.h and
                    MHSGenerator.min_position.x <= x_pos <= MHSGenerator.max_position.x and
                    MHSGenerator.min_position.y <= y_pos <= MHSGenerator.max_position.y and
                    MHSGenerator.min_position.r <= r <= MHSGenerator.max_position.r and
                    valid):
                    out["F"] = [0]  # Valid coordinates
                else:
                    out["F"] = [1]  # Invalid coordinates

        problem = ObstacleProblem()

        algorithm = MixedVariableGA(pop_size=1, n_offsprings=1, survival=RankAndCrowdingSurvival())

        termination = get_termination("time", "00:00:01")  # 1 second

        res = minimize(problem, algorithm, termination, verbose=1)

        print(res.X)
        size = Obstacle.Size(res.X["l"], res.X["w"], res.X["h"])
        position = Obstacle.Position(res.X["x"], res.X["y"], 0, res.X["r"])
        obstacle = Obstacle(size, position)
        test = TestCase(self.case_study, [obstacle])
        try:
            test.execute()
            distances = test.get_distances()
            print(f"minimum_distance:{min(distances)}")
            test.plot()
            test_cases.append(test)
        except Exception as e:
            print("Exception during test execution, skipping the test")
            print(e)

        # for i in range(budget):
        #     size = Obstacle.Size(
        #         l=random.uniform(self.min_size.l, self.max_size.l),
        #         w=random.uniform(self.min_size.w, self.max_size.w),
        #         h=random.uniform(self.min_size.h, self.max_size.h),
        #     )
        #     position = Obstacle.Position(
        #         x=random.uniform(self.min_position.x, self.max_position.x),
        #         y=random.uniform(self.min_position.y, self.max_position.y),
        #         z=0,  # obstacles should always be place on the ground
        #         r=random.uniform(self.min_position.r, self.max_position.r),
        #     )
        #     obstacle = Obstacle(size, position)
        #     test = TestCase(self.case_study, [obstacle])
        #     try:
        #         test.execute()
        #         distances = test.get_distances()
        #         print(f"minimum_distance:{min(distances)}")
        #         test.plot()
        #         test_cases.append(test)
        #     except Exception as e:
        #         print("Exception during test execution, skipping the test")
        #         print(e)

        ### You should only return the test cases
        ### that are needed for evaluation (failing or challenging ones)
        return test_cases


if __name__ == "__main__":
    generator = MHSGenerator("case_studies/mission1.yaml")
    generator.generate(1)
