import random
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.trajectory import Trajectory
from aerialist.px4.position import Position
from testcase import TestCase
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
import math
from pymoo.core.variable import Real, Integer
from shapely.geometry import Polygon

NUM_OBJECTIVES = 4
SPEED = 100

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
        drone_test = DroneTest.from_yaml(case_study_file)
        # TODO: Check if there is a way to speed up the simulation
        drone_test.test.speed = SPEED
        drone_test.simulation.speed = SPEED
        self.case_study = drone_test

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []

        default_test = TestCase(self.case_study, [])
        default_test.test.speed = SPEED
        default_test.test.simulation.speed = SPEED
        print("Executing mission without obstacle")
        try:
            default_test.execute()
            print("Finished")
        except Exception as e:
            print("Exception during test execution, skipping the test")
            print(e)

        class ObstacleProblem(Problem):
            def __init__(self):
                vars = {}
                vars["n"] = Integer(bounds=(1, 3))
                for i in range(3):
                    vars[f"l{i}"] = Real(bounds=(MHSGenerator.min_size.l, MHSGenerator.max_size.l))
                    vars[f"w{i}"] = Real(bounds=(MHSGenerator.min_size.w, MHSGenerator.max_size.w))
                    vars[f"h{i}"] = Real(bounds=(MHSGenerator.min_size.h, MHSGenerator.max_size.h))
                    vars[f"x{i}"] = Real(bounds=(MHSGenerator.min_position.x, MHSGenerator.max_position.x))
                    vars[f"y{i}"] = Real(bounds=(MHSGenerator.min_position.y, MHSGenerator.max_position.y))
                    vars[f"r{i}"] = Real(bounds=(MHSGenerator.min_position.r, MHSGenerator.max_position.r))
                super().__init__(vars=vars, n_obj=NUM_OBJECTIVES)

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
                    o.polygon = Polygon(o.corners)

                    obstacles.append(o)

                # Check if all corners are within the allowed bounds after rotation
                # The output is the number of invalid obstacles, which we want to minimize (0)
                num_out_of_bound = sum(1 for i in range(num_obstacles) if not all(
                    MHSGenerator.min_position.x <= x <= MHSGenerator.max_position.x and
                    MHSGenerator.min_position.y <= y <= MHSGenerator.max_position.y
                    for x, y in obstacles[i].corners
                ))

                # Check for overlapping obstacles
                num_overlapping = sum(1
                                      for i in range(num_obstacles)
                                      for j in range(i + 1, num_obstacles)
                                      if obstacles[i].polygon.intersects(obstacles[j].polygon))
                
                if (num_out_of_bound + num_overlapping) > 0:
                    out["F"] = [float('inf'), float('inf'), float('inf'), float('inf')]
                    return
                
                # Path is feasible
                # TODO: This might be a bit too strong. Should we just leave it to test execution?
                num_too_close = sum(1
                                    for i in range(num_obstacles)
                                    for j in range(i + 1, num_obstacles)
                                    if obstacles[i].polygon.distance(obstacles[j].polygon) < 3)
                
                # At least one obstacle blocks some point of the default trajectory
                min_distance_obstacle_to_trajectory = min(o.polygon.distance(default_test.trajectory.to_line()) for o in obstacles)
                # This will be 0 if the trajectory crosses any obstacle

                # Obstacles should be close to each other
                sum_min_distance_between_obstacles = 0 if len(obstacles) == 1 else sum(
                    obstacles[i].polygon.distance(obstacles[j].polygon)
                    for i in range(num_obstacles)
                    for j in range(i + 1, num_obstacles)
                )

                out["F"] = [
                    num_too_close, 
                    min_distance_obstacle_to_trajectory,
                    sum_min_distance_between_obstacles,
                    num_obstacles
                ]

        problem = ObstacleProblem()

        algorithm = MixedVariableGA(pop_size=1, n_offsprings=1, survival=RankAndCrowdingSurvival())

        termination = get_termination("time", "00:00:20")  # 20 seconds

        res = minimize(problem, algorithm, termination, verbose=1)

        print(res.X)
        obstacles = []
        num_obstacles = res.X[0]["n"]
        for i in range(num_obstacles):
            size = Obstacle.Size(res.X[0][f"l{i}"], res.X[0][f"w{i}"], res.X[0][f"h{i}"])
            position = Obstacle.Position(res.X[0][f"x{i}"], res.X[0][f"y{i}"], 0, res.X[0][f"r{i}"])
            obstacle = Obstacle(size, position)
            obstacles.append(obstacle)

        # Visualize the obstacles without trajectory
        Trajectory(positions = [Position(0, 0, 0, 0, 0), Position(0, 0, 0, 0, 0)]) \
            .plot(obstacles = obstacles)

        test = TestCase(self.case_study, obstacles)
        test.test.speed = SPEED
        test.test.simulation.speed = SPEED
        try:
            print("Executing mission with X = " + str(res.X[0]))
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
