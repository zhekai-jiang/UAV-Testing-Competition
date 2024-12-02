from typing import List, Tuple, Dict
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.trajectory import Trajectory
from aerialist.px4.position import Position
from testcase import TestCase
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
import math
from pymoo.core.variable import Real, Integer, BoundedVariable
from shapely.geometry import Polygon, Point
import csv

NUM_OBJECTIVES: int = 5
SPEED: int = 100
all_runs: List[Tuple[TestCase, float]] = []
test_cases: List[TestCase] = []

class ObstacleParams:
    def __init__(self, x: Dict[str, float], i: int) -> None:
        self.l = x[f"l{i}"]
        self.w = x[f"w{i}"]
        self.h = x[f"h{i}"]
        self.x = x[f"x{i}"]
        self.y = x[f"y{i}"]
        self.r = x[f"r{i}"]
        self.polygon: Polygon = None

params_keys = ["n"] + [f"{p}{i}" for p in ["l", "w", "h", "x", "y", "r"] for i in range(3)]

class ObstaclePlacementProblem(ElementwiseProblem):
    def __init__(self, case_study: TestCase, starting_point: Point):
        self.case_study = case_study
        self.starting_point = starting_point
        vars: Dict[str, BoundedVariable] = {}
        vars["n"] = Integer(bounds=(2, 3)) # TODO: 1 may not be good for now
        for i in range(3):
            vars[f"l{i}"] = Real(bounds=(MHSGenerator.min_size.l, MHSGenerator.max_size.l))
            vars[f"w{i}"] = Real(bounds=(MHSGenerator.min_size.w, MHSGenerator.max_size.w))
            vars[f"h{i}"] = Real(bounds=(MHSGenerator.min_size.h, MHSGenerator.max_size.h))
            vars[f"x{i}"] = Real(bounds=(MHSGenerator.min_position.x, MHSGenerator.max_position.x))
            vars[f"y{i}"] = Real(bounds=(MHSGenerator.min_position.y, MHSGenerator.max_position.y))
            vars[f"r{i}"] = Real(bounds=(MHSGenerator.min_position.r, MHSGenerator.max_position.r))
        super().__init__(vars=vars, n_obj=NUM_OBJECTIVES)

    def _evaluate(self, x: Dict[str, float], out: Dict[str, List[float]], *args, **kwargs):
        num_obstacles = x["n"]

        obstacle_params: List[ObstacleParams] = []

        for i in range(num_obstacles):
            o = ObstacleParams(x, i)

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

            obstacle_params.append(o)

        # Check if all corners are within the allowed bounds after rotation
        # The output is the number of invalid obstacles, which we want to minimize (0)
        num_out_of_bound = sum(1 for i in range(num_obstacles) if not all(
            MHSGenerator.min_position.x <= x <= MHSGenerator.max_position.x and
            MHSGenerator.min_position.y <= y <= MHSGenerator.max_position.y
            for x, y in obstacle_params[i].corners
        ))
        # TODO: Use distance

        # Check for overlapping obstacles
        num_overlapping = sum(1
                            for i in range(num_obstacles)
                            for j in range(i + 1, num_obstacles)
                            if obstacle_params[i].polygon.intersects(obstacle_params[j].polygon))
        # TODO: Area of intersection
        # shapely p.intersection(q).area
        # Maybe: something like AABB to approximate -> https://docs.scenic-lang.org/en/1.x/_modules/scenic/core/regions.html#Region.getAABB
        
        # if (num_out_of_bound + num_overlapping) > 0:
            # out["F"] = [float('inf'), float('inf'), float('inf'), float('inf')]
            # return
        
        # Path is feasible
        # TODO: This might be a bit too strong. Should we just leave it to test execution?
        num_too_close = sum(1
                            for i in range(num_obstacles)
                            for j in range(i + 1, num_obstacles)
                            if obstacle_params[i].polygon.distance(obstacle_params[j].polygon) < 3)
        
        # At least one obstacle blocks some point of the default trajectory
        min_distance_obstacle_to_trajectory = sum(min(o.polygon.distance(test.trajectory.to_line()) for o in obstacle_params) for (test, _) in all_runs)
        # This will be 0 if the trajectory crosses any obstacle

        # Obstacles should be close to each other
        max_min_distance_to_other_obstacles = \
            0 if len(obstacle_params) == 1 \
            else max(
                min(obstacle_params[i].polygon.distance(obstacle_params[j].polygon)
                    for j in range(num_obstacles) if j != i
                )
                for i in range(num_obstacles)
            )

        # There should be something close to the starting point
        min_distance_to_start = min([self.starting_point.distance(o.polygon) for o in obstacle_params])

        if num_out_of_bound + num_overlapping + num_too_close + min_distance_obstacle_to_trajectory > 1e-5 \
            or max_min_distance_to_other_obstacles > 5:
            out["F"] = [
                num_out_of_bound + num_overlapping + num_too_close, 
                min_distance_obstacle_to_trajectory if num_out_of_bound + num_overlapping + num_too_close == 0 else float('inf'),
                max_min_distance_to_other_obstacles if num_out_of_bound + num_overlapping + num_too_close + min_distance_obstacle_to_trajectory < 1e-5 else float('inf'),
                float('inf'),# min_distance_to_start,
                # num_obstacles,
                float('inf')
            ]
            print([x[k] for k in params_keys])
            print([num_out_of_bound, num_overlapping, num_too_close, min_distance_obstacle_to_trajectory, max_min_distance_to_other_obstacles])
        else:
            print("Executing mission with X = " + str(x))
            obstacles_test = [Obstacle(Obstacle.Size(o.l, o.w, o.h), Obstacle.Position(o.x, o.y, 0, o.r)) for o in obstacle_params]
            test = TestCase(self.case_study, obstacles_test)
            Trajectory(positions = [Position(0, 0, 0, 0, 0), Position(0, 0, 0, 0, 0)]) \
                .plot(obstacles = obstacles_test)
            test.test.speed = SPEED
            test.test.simulation.speed = SPEED
            min_distance = float('inf')
            try:
                test.execute()
                distances = test.get_distances()
                min_distance = min(distances)
                print(f"minimum_distance:{min_distance}")
                test.plot()
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)
            result = 0 if min_distance < 0.25 \
                else 3 if min_distance < 1 \
                else 4 if min_distance < 1.5 \
                else min_distance + 5
            all_runs.append((test, result))
            test_cases.append(test)
            with open('results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(params_keys + ["min_distance", "result"])
                writer.writerow([x[k] for k in params_keys] + [min_distance, result])
                print([x[k] for k in params_keys])
                print([num_out_of_bound, num_overlapping, num_too_close, min_distance_obstacle_to_trajectory, max_min_distance_to_other_obstacles, min_distance, result])

            out["F"] = [
                num_out_of_bound + num_overlapping + num_too_close, 
                min_distance_obstacle_to_trajectory,
                max_min_distance_to_other_obstacles,
                min_distance_to_start,
                # num_obstacles,
                result
            ]

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
        default_test = TestCase(self.case_study, [])
        default_test.test.speed = SPEED
        default_test.test.simulation.speed = SPEED
        print("Executing mission without obstacle")
        try:
            default_test.execute()
            print("Finished")
            all_runs.append((default_test, float('inf')))
        except Exception as e:
            print("Exception during test execution, skipping the test")
            print(e)

        problem = ObstaclePlacementProblem(self.case_study, Point(default_test.trajectory.to_line().coords[0]))

        algorithm = MixedVariableGA(pop_size=5, n_offsprings=5, survival=RankAndCrowding())

        termination = get_termination("time", "01:30:00")  # hours:minutes:seconds

        res = minimize(problem, algorithm, termination, verbose=1)

        ### You should only return the test cases
        ### that are needed for evaluation (failing or challenging ones)
        return test_cases


if __name__ == "__main__":
    generator = MHSGenerator("case_studies/mission1.yaml")
    generator.generate(2)
