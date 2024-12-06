from typing import List, Tuple, Dict
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.trajectory import Trajectory
from aerialist.px4.position import Position
from testcase import TestCase
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.core.termination import Termination
from pymoo.core.problem import ElementwiseProblem
import math
from pymoo.core.variable import Real, Integer, BoundedVariable
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
from shapely.affinity import rotate
import csv
from itertools import chain
import time

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.max_time import TimeBasedTermination


NUM_OBJECTIVES: int = 8
SPEED: int = 100

min_size = Obstacle.Size(2, 2, 10)
max_size = Obstacle.Size(20, 20, 25)
min_position = Obstacle.Position(-40, 10, 0, 0)
max_position = Obstacle.Position(30, 40, 0, 90)

class Execution:
    def __init__(self, testCase: TestCase, depth: int, result: float) -> None:
        self.testCase = testCase
        self.result = result
        self.depth = depth
        self.followup: List[Execution] = []

all_executions: List[Execution] = []
test_cases: List[TestCase] = []

def denormalize(v, min_val, max_val):
    val =  min_val + (max_val-min_val)*v
    return val

class ObstacleParams:
    def __init__(self, x: Dict[str, float], i: int) -> None:
        # self.l = x[f"l{i}"]
        # self.w = x[f"w{i}"]
        # self.h = x[f"h{i}"]
        # self.x = x[f"x{i}"]
        # self.y = x[f"y{i}"]
        # self.r = x[f"r{i}"]
        self.l = denormalize(x[i*6+0], min_size.l, max_size.l)
        self.w = denormalize(x[i*6+1], min_size.w, max_size.w)
        self.h = denormalize(x[i*6+2], min_size.h, max_size.h)
        self.x = denormalize(x[i*6+3], min_position.x, max_position.x)
        self.y = denormalize(x[i*6+4], min_position.y, max_position.y)
        self.r = denormalize(x[i*6+5], min_position.r, max_position.r)
        self.polygon: Polygon = None
        self.corners: List[Tuple[float, float]] = None

params_keys = ["n"] + [f"{p}{i}" for p in ["l", "w", "h", "x", "y", "r"] for i in range(3)]

class ObstaclePlacementProblem(ElementwiseProblem):
    def __init__(self, case_study: TestCase, starting_point: Point):
        self.case_study = case_study
        self.starting_point = starting_point
        # vars: Dict[str, BoundedVariable] = {}
        # vars["n"] = Integer(bounds=(1, 3))
        # for i in range(2):
        #     vars[f"l{i}"] = Real(bounds=(min_size.l, max_size.l))
        #     vars[f"w{i}"] = Real(bounds=(min_size.w, max_size.w))
        #     vars[f"h{i}"] = Real(bounds=(min_size.h, max_size.h))
        #     vars[f"x{i}"] = Real(bounds=(min_position.x, max_position.x))
        #     vars[f"y{i}"] = Real(bounds=(min_position.y, max_position.y))
        #     vars[f"r{i}"] = Real(bounds=(min_position.r, max_position.r))
        # super().__init__(vars=vars, n_obj=NUM_OBJECTIVES)
        super().__init__(n_var=12, n_obj=NUM_OBJECTIVES, n_constr=0, xl=0.0, xu=1.0)


    def _evaluate(self, x: Dict[str, float], out: Dict[str, List[float]], *args, **kwargs):
        # num_obstacles = x["n"]
        num_obstacles = 2
        # print(x)

        # We want to put one obstacle at a time
        # max_num_obstacles = max(e.depth + 1 for e in all_executions)

        # if num_obstacles > max_num_obstacles:
        #     out["F"] = [num_obstacles - max_num_obstacles] + [float('inf')] * (NUM_OBJECTIVES - 1)
        #     return

        obstacle_params: List[ObstacleParams] = []

        for i in range(num_obstacles):
            o = ObstacleParams(x, i)

            # Calculate the rotated bounds for all four corners of the rectangle
            half_l = o.l / 2
            half_w = o.w / 2
            o.polygon = Polygon([(o.x - half_l, o.y - half_w), (o.x + half_l, o.y - half_w), (o.x + half_l, o.y + half_w), (o.x - half_l, o.y + half_w)])
            o.polygon = rotate(o.polygon, o.r)
            # There was some weird issue with invalidity of geometry while taking intersection somehow.
            # Now I use the shapely methods above instead of coding the transformations in triangular functions myself.
            # In case this doesn't work still, the thing below may be a fix.
            # if not o.polygon.is_valid:
            #     o.polygon = make_valid(o.polygon)

            obstacle_params.append(o)

        # print("here")
        # Check if all corners are within the allowed bounds after rotation
        # The output is the sum of distances that exceed the boundary, which we want to minimize (0)
        distances_out_of_bound = [sum(max(0, min_position.x - x) + max(0, x - max_position.x) + max(0, min_position.y - y) + max(0, y - max_position.y) for x, y in o.polygon.exterior.coords[:4])
                                  for o in obstacle_params]

        # Check for overlapping obstacles
        areas_overlapping = [sum(o.polygon.intersection(p.polygon).area for p in obstacle_params if p != o)
                             for o in obstacle_params]
        
        # To try to ensure the mission is feasible
        sums_dists_to_others = [sum(o.polygon.distance(p.polygon) for p in obstacle_params if p != o)
                                for o in obstacle_params]
        
        # Obstacles should block at least one trajectory at their depth, determined also by the previous obstacles
        min_distances_obstacle_to_trajectory: List[float] = []
        # execs_at_depth = set([e for e in all_executions if e.depth == 0])
        execs_at_depth = all_executions
        parent_execs = set()
        for depth in range(num_obstacles):
            # This will be 0 if the trajectory crosses the obstacle
            min_distances_obstacle_to_trajectory.append(min(obstacle_params[depth].polygon.distance(e.testCase.trajectory.to_line()) for e in execs_at_depth))
            parent_execs = execs_at_depth
            # execs_at_depth = set(chain.from_iterable(e.followup for e in execs_at_depth))

        # Obstacles should be close to each other
        # max_min_distance_to_other_obstacles = \
        #     0 if len(obstacle_params) == 1 \
        #     else max(
        #         min(obstacle_params[i].polygon.distance(obstacle_params[j].polygon)
        #             for j in range(num_obstacles) if j != i
        #         )
        #         for i in range(num_obstacles)
        #     )

        # There should be something close to the starting point
        min_distance_to_start = min([self.starting_point.distance(o.polygon) for o in obstacle_params])

        enforced_heuristics = [distances_out_of_bound[i] + areas_overlapping[i] + min_distances_obstacle_to_trajectory[i] for i in range(num_obstacles)]

        enforced_heuristics += [sum(enforced_heuristics) / num_obstacles] * (3 - num_obstacles)

        if sum(enforced_heuristics) > 1e-7: # \
            # or max_min_distance_to_other_obstacles > 10:
            # out["F"] = [
            #     0,
            #     enforced_heuristics[0],
            #     enforced_heuristics[1],
            #     enforced_heuristics[2],
            #     # sqrt to make it less important, minimize negation to maximize actual distance
            #     -math.sqrt(sums_dists_to_others[0]),
            #     -math.sqrt(sums_dists_to_others[1]),
            #     -math.sqrt(sums_dists_to_others[2]),
            #     # max_min_distance_to_other_obstacles if sum_distance_out_of_bound + area_overlapping + num_too_close + sum_min_distance_obstacle_to_trajectory < 1e-5 else float('inf'),
            #     min_distance_to_start,
            #     num_obstacles,
            #     float('inf')
            # ]
            out["F"] = [
                0,
                enforced_heuristics[0],
                enforced_heuristics[1],
                # enforced_heuristics[2],
                # sqrt to make it less important, minimize negation to maximize actual distance
                -math.sqrt(sums_dists_to_others[0]),
                -math.sqrt(sums_dists_to_others[1]),
                # -math.sqrt(sums_dists_to_others[2]),
                # max_min_distance_to_other_obstacles if sum_distance_out_of_bound + area_overlapping + num_too_close + sum_min_distance_obstacle_to_trajectory < 1e-5 else float('inf'),
                min_distance_to_start,
                num_obstacles,
                100000
            ]
            # print([x[k] for k in params_keys])
            # print(x)
            # print("---")
            # print("Distances out of bound " + str(distances_out_of_bound))
            # print("Areas overlapping " + str(areas_overlapping))
            # print("Min distances to trajectory " + str(min_distances_obstacle_to_trajectory))
            # print("Min distances to others " + str(sums_dists_to_others))
            # print("Min distance to starting point " + str(min_distance_to_start))

            # print(out["F"])
        else:
            print("Executing mission with X = " + str(x))
            obstacles_test = [Obstacle(Obstacle.Size(o.l, o.w, o.h), Obstacle.Position(o.x, o.y, 0, o.r)) for o in obstacle_params]
            test = TestCase(self.case_study, obstacles_test)
            # Trajectory(positions = [Position(0, 0, 0, 0, 0), Position(0, 0, 0, 0, 0)]) \
            #     .plot(obstacles = obstacles_test)
            # time.sleep(1)
            test.test.speed = SPEED
            test.test.simulation.speed = SPEED
            min_distance = float('inf')
            try:
                test.execute()
                # TODO: Check for timeout?
                distances = test.get_distances()
                min_distance = min(distances)
                print(f"minimum_distance:{min_distance}")
                test.plot()
                time.sleep(1)
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)
            result = 0 if min_distance < 0.25 \
                else 3 if min_distance < 1 \
                else 4 if min_distance < 1.5 \
                else min_distance + 5
            e = Execution(test, num_obstacles, result)
            for parent in parent_execs:
                parent.followup.append(e)
            all_executions.append(e)
            test_cases.append(test)
            with open('results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(params_keys + ["min_distance", "result"])
                writer.writerow([x[k] for k in range(12)] + [min_distance, result])
                # writer.writerow([x[k] for k in params_keys] + [min_distance, result])
            # print([x[k] for k in params_keys])
            # print(x)
            print("---")
            print("Distances out of bound " + str(distances_out_of_bound))
            print("Areas overlapping " + str(areas_overlapping))
            print("Min distances to trajectory " + str(min_distances_obstacle_to_trajectory))
            print("Min distances to others " + str(sums_dists_to_others))
            print("Min distance to starting point " + str(min_distance_to_start))

            # out["F"] = [
            #     0,
            #     enforced_heuristics[0],
            #     enforced_heuristics[1],
            #     enforced_heuristics[2],
            #     # sqrt to make it less important, minimize negation to maximize actual distance
            #     -math.sqrt(sums_dists_to_others[0]),
            #     -math.sqrt(sums_dists_to_others[1]),
            #     -math.sqrt(sums_dists_to_others[2]),
            #     # max_min_distance_to_other_obstacles if sum_distance_out_of_bound + area_overlapping + num_too_close + sum_min_distance_obstacle_to_trajectory < 1e-5 else float('inf'),
            #     min_distance_to_start,
            #     num_obstacles,
            #     result ** 3 # Make it super important
            # ]
            out["F"] = [
                0,
                enforced_heuristics[0],
                enforced_heuristics[1],
                # enforced_heuristics[2],
                # sqrt to make it less important, minimize negation to maximize actual distance
                -math.sqrt(sums_dists_to_others[0]),
                -math.sqrt(sums_dists_to_others[1]),
                # -math.sqrt(sums_dists_to_others[2]),
                # max_min_distance_to_other_obstacles if sum_distance_out_of_bound + area_overlapping + num_too_close + sum_min_distance_obstacle_to_trajectory < 1e-5 else float('inf'),
                min_distance_to_start,
                num_obstacles,
                result ** 3 # Make it super important
            ]

            # print(out["F"])
class MHSGenerator(object):

    def __init__(self, case_study_file: str) -> None:
        drone_test = DroneTest.from_yaml(case_study_file)
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
            # default_test.trajectory = Trajectory(positions = [Position(0, 0, 0, 0, 0), Position(0, 0, 0, 0, 0)])
            default_test.plot()
            time.sleep(1)
            print("Finished")
            all_executions.append(Execution(default_test, 0, float('inf')))
        except Exception as e:
            print("Exception during test execution, skipping the test")
            print(e)
            exit()

        problem = ObstaclePlacementProblem(self.case_study, Point(default_test.trajectory.to_line().coords[0]))

        # algorithm = MixedVariableGA(pop_size=50, n_offsprings=100, survival=RankAndCrowding())
        ref_dirs = get_reference_directions("das-dennis", n_dim=NUM_OBJECTIVES, n_partitions=1)
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=50, n_offsprings=100, eliminate_duplicates=True)
        # algorithm = NSGA2(pop_size=50, n_offsprings=100)


        class ObstaclePlacementTermination(Termination):
            def _update(self, algorithm):
                r = 1 if len(all_executions) >= budget else 0
                # print(r)
                return r
                return (len(all_executions)+1)/(budget+1)

        termination = ObstaclePlacementTermination()


        res = minimize(problem, algorithm, termination, verbose=1)

        ### You should only return the test cases
        ### that are needed for evaluation (failing or challenging ones)
        return test_cases


if __name__ == "__main__":
    generator = MHSGenerator("case_studies/mission1.yaml")
    generator.generate(200)
