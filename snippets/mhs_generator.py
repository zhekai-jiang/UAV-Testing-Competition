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
from pymoo.core.variable import Real, Integer

class ObstacleParams:
    def __init__(self, x, i):
        self.l = x[f"l{i}"]
        self.w = x[f"w{i}"]
        self.h = x[f"h{i}"]
        self.x = x[f"x{i}"]
        self.y = x[f"y{i}"]
        self.r = x[f"r{i}"]

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
                vars["n"] = Integer(bounds=(1, budget))
                for i in range(3):
                    vars[f"l{i}"] = Real(bounds=(MHSGenerator.min_size.l, MHSGenerator.max_size.l))
                    vars[f"w{i}"] = Real(bounds=(MHSGenerator.min_size.w, MHSGenerator.max_size.w))
                    vars[f"h{i}"] = Real(bounds=(MHSGenerator.min_size.h, MHSGenerator.max_size.h))
                    vars[f"x{i}"] = Real(bounds=(MHSGenerator.min_position.x, MHSGenerator.max_position.x))
                    vars[f"y{i}"] = Real(bounds=(MHSGenerator.min_position.y, MHSGenerator.max_position.y))
                    vars[f"r{i}"] = Real(bounds=(MHSGenerator.min_position.r, MHSGenerator.max_position.r))
                super().__init__(vars=vars, n_obj=1)

            def _evaluate(self, x, out, *args, **kwargs):
                num_obstacles = x[0]["n"]

                obstacles = []

                for i in range(num_obstacles):
                    o = ObstacleParams(x[0], i)

                    # Calculate the rotated bounds for all four corners of the rectangle
                    half_l = o.l / 2
                    half_w = o.w / 2
                    o.corners = [
                        (o.x + half_l * math.cos(math.radians(o.r)) - half_w * math.sin(math.radians(o.r)),
                            o.y + half_l * math.sin(math.radians(o.r)) + half_w * math.cos(math.radians(o.r))),
                        (o.x - half_l * math.cos(math.radians(o.r)) - half_w * math.sin(math.radians(o.r)),
                            o.y - half_l * math.sin(math.radians(o.r)) + half_w * math.cos(math.radians(o.r))),
                        (o.x + half_l * math.cos(math.radians(o.r)) + half_w * math.sin(math.radians(o.r)),
                            o.y + half_l * math.sin(math.radians(o.r)) - half_w * math.cos(math.radians(o.r))),
                        (o.x - half_l * math.cos(math.radians(o.r)) + half_w * math.sin(math.radians(o.r)),
                            o.y - half_l * math.sin(math.radians(o.r)) - half_w * math.cos(math.radians(o.r)))
                    ]

                    obstacles.append(o)

                # Check if all corners are within the allowed bounds after rotation
                # The output is the number of invalid obstacles, which we want to minimize (0)
                num_invalid = sum(1 for i in range(num_obstacles) if not all(
                    MHSGenerator.min_position.x <= x <= MHSGenerator.max_position.x and
                    MHSGenerator.min_position.y <= y <= MHSGenerator.max_position.y
                    for x, y in obstacles[i].corners
                ))

                out["F"] = [num_invalid]

        problem = ObstacleProblem()

        algorithm = MixedVariableGA(pop_size=1, n_offsprings=1, survival=RankAndCrowdingSurvival())

        termination = get_termination("time", "00:00:01")  # 1 second

        res = minimize(problem, algorithm, termination, verbose=1)

        print(res.X)
        obstacles = []
        num_obstacles = res.X["n"]
        for i in range(num_obstacles):
            size = Obstacle.Size(res.X[f"l{i}"], res.X[f"w{i}"], res.X[f"h{i}"])
            position = Obstacle.Position(res.X[f"x{i}"], res.X[f"y{i}"], 0, res.X[f"r{i}"])
            obstacle = Obstacle(size, position)
            obstacles.append(obstacle)
        test = TestCase(self.case_study, obstacles)
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
    generator.generate(2)
