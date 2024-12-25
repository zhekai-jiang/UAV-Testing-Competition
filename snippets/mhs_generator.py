from typing import List, Tuple, Dict
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
import numpy as np
from testcase import TestCase
from pymoo.optimize import minimize
from pymoo.core.termination import Termination
from pymoo.core.problem import ElementwiseProblem
import math
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
import csv
import time

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3


NUM_OBJECTIVES: int = 3
NUM_VARIABLES: int = 10
SPEED: int = 100

min_size = Obstacle.Size(7, 10, 15)
max_size = Obstacle.Size(20, 20, 15)
min_position = Obstacle.Position(-40, 10, 0, 0)
max_position = Obstacle.Position(30, 40, 0, 90)
height = 15

DIST_TOLERANCE = 5 # TODO MAYBE optimize these values?
DIST_OBS_MAX = 16
DIST_OBS_MIN = 8
DIST_START_TOLERANCE = 5
SCENARIO_SIMILARITY_THRESHOLD = 0.2 # TODO Test this out

class Execution:
    def __init__(self, params_vector: List[float], testCase: TestCase, min_distance: float, result: float) -> None:
        self.params_vector = params_vector
        self.testCase = testCase
        self.min_distance = min_distance
        self.result = result

BUDGET: int = 200
all_executions: List[Execution] = []
initial_execution: List[Execution] = []
good_single_obstacle_tests: List[TestCase] = []
test_cases: List[TestCase] = []

initial_population_fixed = [
    [0.5248214908364522,0.1728250063428862,0.380872169470668,0.32746438646966014, 0.4224165503721484, 0.0633264672176352,0.7030562867295299,0.7265519774157527,0.4427178720414508, 0.00390712094969714], # 0
    [0.002437904, 0.044083212040659236, 0.38854313448953864, 0.22548564372189597, 0.7049354162940153, 0.095829846, 0.071981721, 0.5513957238231902, 0.7249793156058763, 0.9867444951163424], # 0.66
    [0.028888049824159157,0.07755462712603683,0.41712032578357994,0.2283720200961537, 0.7486728307315297, 0.06165770740907471,0.873677211665613,0.6415718765772439,0.6616778443126642, 0.8025448773926038], # 0.91
    [0.028888049824159157,0.07755462712603683,0.41712032578357994,0.6616778443126642, 0.7486728307315297,
     0.06165770740907471,0.873677211665613,0.6415718765772439,    0.2283720200961537, 0.8025448773926038],
    [0.516779953730593,0.6168793554484786,0.409314811389346,0.32219672414981126, 0.13530056143545108, 0.11383979587593976,0.7009896726535482,0.718787611005901,0.8535107450924464, 0.966362157736934], # 1.11
    [0.5167791040739792,0.6168720588870445,0.4091818198611974,0.3287955613619398, 0.7473642659997762, 0.00312072688042582,0.7009896726535482,0.7188264559851381,0.8583628486323605, 0.9660296547041434], # 1.45
]

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
        self.l = denormalize(x[i*5+0], min_size.l, max_size.l)
        self.w = denormalize(x[i*5+1], min_size.w, max_size.w)
        # self.h = denormalize(x[i*6+2], min_size.h, max_size.h)
        self.x = denormalize(x[i*5+2], min_position.x, max_position.x)
        self.y = denormalize(x[i*5+3], min_position.y, max_position.y)
        self.r = denormalize(x[i*5+4], min_position.r, max_position.r)
        self.polygon: Polygon = None
        self.corners: List[Tuple[float, float]] = None

# params_keys = ["n"] + [f"{p}{i}" for p in ["l", "w", "h", "x", "y", "r"] for i in range(3)]
params_keys = [f"{p}{i}" for p in ["l", "w", "x", "y", "r"] for i in range(2)]

class ObstaclePlacementProblem(ElementwiseProblem):
    def __init__(self, case_study: TestCase, starting_point: Point):
        self.case_study = case_study
        self.starting_point = starting_point

        
        # Get closest dist to validity space
        x__, y__ = starting_point.x, starting_point.y
        dx_ = 0 if min_position.x <= x__ <= max_position.x else max(min_position.x - x__, x__-max_position.x)
        dy_ = 0 if min_position.y <= y__ <= max_position.y else max(min_position.y - y__, y__-max_position.y)
        self.dist_start_tolerance = math.sqrt(dx_**2+dy_**2) + DIST_START_TOLERANCE

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
        super().__init__(n_var=NUM_VARIABLES, n_obj=NUM_OBJECTIVES, n_constr=0, xl=0.0, xu=1.0)


    def _evaluate(self, x: List[float], out: Dict[str, List[float]], *args, **kwargs):
        global BUDGET
        global all_executions

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
        # distances_out_of_bound = [sum(max(0, min_position.x - x) + max(0, x - max_position.x) + max(0, min_position.y - y) + max(0, y - max_position.y) for x, y in o.polygon.exterior.coords[:4])
        #                           for o in obstacle_params]
        # distances_out_of_bound = [sum(max(0, min_position.x - x) + max(0, x - max_position.x) + max(0, min_position.y - y) + max(0, y - max_position.y) for x, y in o.polygon.exterior.coords[:4])
        #                           for o in obstacle_params]

        #####################
        distances_out_of_bound = []
        for o in obstacle_params:
            max_d = 0
            for x_, y_ in o.polygon.exterior.coords[:4]:
                dx = 0 if min_position.x <= x_ <= max_position.x else max(min_position.x - x_, x_-max_position.x)
                dy = 0 if min_position.y <= y_ <= max_position.y else max(min_position.y - y_, y_-max_position.y)
                d = math.sqrt(dx**2+dy**2)
                if d > max_d:
                    max_d = d
            distances_out_of_bound.append(max_d)

        oob_1 = distances_out_of_bound[0]
        oob_2 = distances_out_of_bound[1]
        # NOTE this assumes that the UAV will indeed cross through the valid obstacle placement space

        #####################
        # Check for overlapping obstacles
        # areas_overlapping = [sum(o.polygon.intersection(p.polygon).area for p in obstacle_params if p != o)
                            #  for o in obstacle_params]
        areas_overlapping = obstacle_params[0].polygon.intersection(obstacle_params[1].polygon).area
        # ovr_1 = areas_overlapping[0]
        ovr_2 = areas_overlapping

        #####################
        # To try to ensure the mission is feasible
        sums_dists_to_others = [sum(o.polygon.distance(p.polygon) for p in obstacle_params if p != o)
                                for o in obstacle_params]
        # dis_oth_1 = sums_dists_to_others[0]
        dis_oth_2 = sums_dists_to_others[1]

        #####################
        # # Obstacles should block at least one trajectory at their depth, determined also by the previous obstacles
        # min_distances_obstacle_to_trajectory: List[float] = []
        # # execs_at_depth = set([e for e in all_executions if e.depth == 0])
        # execs_at_depth = all_executions
        # parent_execs = set()
        # for depth in range(num_obstacles):
        #     # This will be 0 if the trajectory crosses the obstacle
        #     min_distances_obstacle_to_trajectory.append(min(obstacle_params[depth].polygon.distance(e.testCase.trajectory.to_line()) for e in execs_at_depth))
        #     parent_execs = execs_at_depth
        #     # execs_at_depth = set(chain.from_iterable(e.followup for e in execs_at_depth))

        dis_path_1 = obstacle_params[0].polygon.distance(initial_execution[0].testCase.trajectory.to_line())
        if len(all_executions) == 0:
            executions = initial_execution
        else:
            executions = all_executions
        dis_path_2 = min(obstacle_params[1].polygon.distance(e.testCase.trajectory.to_line()) for e in executions) # TODO MAYBE we could improve this, butit is not a priority


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
        dis_start_1 = self.starting_point.distance(obstacle_params[0].polygon)

        # Closer obstacle
        heu_1 = oob_1 + max(0, dis_path_1-DIST_TOLERANCE) + max(0, dis_start_1-self.dist_start_tolerance)
        heu_2 = oob_2 + max(0, dis_path_2-DIST_TOLERANCE) + max(DIST_OBS_MIN-dis_oth_2, dis_oth_2-DIST_OBS_MAX) + math.sqrt(ovr_2)


        # enforced_heuristics = [distances_out_of_bound[i] + areas_overlapping[i] + min_distances_obstacle_to_trajectory[i] for i in range(num_obstacles)]
        # enforced_heuristics += [sum(enforced_heuristics) / num_obstacles] * (3 - num_obstacles)

        # print(oob_1, max(0, dis_path_1-DIST_TOLERANCE), max(0, dis_start_1-DIST_START_TOLERANCE))
        # print(oob_2, max(0, dis_path_2-DIST_TOLERANCE), max(0, dis_oth_2-DIST_OBS_TOLERANCE), math.sqrt(ovr_2))
        # print('---')

        if heu_1+heu_2 > 1e-3 and min(sum(abs(a - b) for a, b in zip(x, known_x)) for known_x in initial_population_fixed) > 1e-3:
            ###############
            # SCENARIO INVALID
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
                heu_1,
                heu_2,
                1000000
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
            ###############
            # SCENARIO VALID - SIMULATE
            
            print("Found valid mission with X = " + str(x))

            # Check if similar scenario has already been run
            cur_encoding =  list(x)
            duplicate_found = False
            for e in all_executions:
                encoding = e.params_vector
                res = e.result
                diff = sum(abs(a - b) for a, b in zip(cur_encoding, encoding))
                if diff < SCENARIO_SIMILARITY_THRESHOLD:
                    duplicate_found = True
                    # TODO This is important: if scenarios become too similar with time, or if the similarity threshold is too big, we may end up in a situation where nothing is getting simulated.
                    # We must avoid this.
                    # Option 1 is to gradually decrease similarity threshold?
                    # Option 2 is to gradually reduce the value of res in the "simulations" list by some percentage for repeating scenarios. This way, from the perspective of the MHS, similar (and thus, unsimulated) scenarios will gradually be "worse" so they will eventually be lost
                    break
            
            # Simulation not run
            if duplicate_found or len(all_executions) + 3 >= BUDGET:
                print("<<<Skipping - Duplicate>>>")
                out["F"] = [
                    heu_1,
                    heu_2,
                    res
                ]
                return

            # RUN SIMULATION
            print("<<<Executing>>>")
            obstacles_test = [Obstacle(Obstacle.Size(float(o.l), float(o.w), height), Obstacle.Position(float(o.x), float(o.y), 0, float(o.r))) for o in obstacle_params]
            test = TestCase(self.case_study, obstacles_test)
            test.test.speed = SPEED
            test.test.simulation.speed = SPEED
            min_distance = float('inf')

            try:
                test.execute()
                distances = test.get_distances()
                min_distance = min(distances)
                print(f"minimum_distance:{min_distance}")
                test.plot()
                time.sleep(1)
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)            
                out["F"] = [
                    heu_1,
                    heu_2,
                    1000000
                ]
                return

            # GATHER RESULTS
            result = min_distance ** 3 \
                + (3 if min_distance >= 0.25 else 0) \
                + 1 * (max(0, min_distance - 0.25) ** 2) \
                + (1 if min_distance >= 1 else 0) \
                + 2 * (max(0, min_distance - 1) ** 2) \
                + (4.5 if min_distance >= 1.5 else 0) \
                + math.log(max(0, min_distance - 1.5) + 1, 1.5)
            # 0 if min_distance < 0.25 \
            #     else 3 if min_distance < 1 \
            #     else 4 if min_distance < 1.5 \
            #     else min_distance + 8.5
            e = Execution(list(x), test, min_distance, result)
            # for parent in parent_execs:
            #     parent.followup.append(e)
            all_executions.append(e)
            test_cases.append(test)

            # SAVE INFO
            with open('results.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(params_keys + ["min_distance", "result"])
                writer.writerow([x[k] for k in range(NUM_VARIABLES)] + [min_distance, result])
                # writer.writerow([x[k] for k in params_keys] + [min_distance, result])
            out["F"] = [
                heu_1,
                heu_2,
                result
            ]
            # print([x[k] for k in params_keys])
            # print(x)
            # print("---")
            # print("Distances out of bound " + str(distances_out_of_bound))
            # print("Areas overlapping " + str(areas_overlapping))
            # print("Min distances to trajectory " + str(min_distances_obstacle_to_trajectory))
            # print("Min distances to others " + str(sums_dists_to_others))
            # print("Min distance to starting point " + str(min_distance_to_start))

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


class MHSGenerator(object):

    def __init__(self, case_study_file: str) -> None:
        drone_test = DroneTest.from_yaml(case_study_file)
        drone_test.test.speed = SPEED
        drone_test.simulation.speed = SPEED
        self.case_study = drone_test

    def generate(self, budget: int) -> List[TestCase]:
        global BUDGET
        global all_executions
        BUDGET = budget
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
            initial_execution.append(Execution([], default_test, float('inf'), float('inf')))
            # all_executions.append(Execution(default_test, 0, float('inf')))
        except Exception as e:
            print("Exception during test execution, skipping the test")
            print(e)
            exit()
        

        single_obstacle_test_cases = [
            TestCase(self.case_study, [Obstacle(Obstacle.Size(20, 12, 20), Obstacle.Position(5.5, 19.5, 0, 20))]),
            TestCase(self.case_study, [Obstacle(Obstacle.Size(12, 20, 20), Obstacle.Position(-5.5, 19.5, 0, 70))])
        ]
        for t in single_obstacle_test_cases:
            try:
                print("Executing mission with one obstacle")
                t.execute()
                t.plot()
                time.sleep(1)
                print("Finished")
                min_distance = min(t.get_distances())
                if min_distance < 1.5:
                    good_single_obstacle_tests.append(t)
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)

        problem = ObstaclePlacementProblem(self.case_study, Point(default_test.trajectory.to_line().coords[0]))

        # algorithm = MixedVariableGA(pop_size=50, n_offsprings=100, survival=RankAndCrowding())
        ref_dirs = get_reference_directions("das-dennis", n_dim=NUM_OBJECTIVES, n_partitions=1)
        initial_population = np.concatenate((initial_population_fixed, np.random.random((50 - len(initial_population_fixed), NUM_VARIABLES))))
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=50, n_offsprings=100, eliminate_duplicates=True, sampling=initial_population)
        # algorithm = NSGA2(pop_size=50, n_offsprings=100)


        class ObstaclePlacementTermination(Termination):
            def _update(self, algorithm):
                r = 1 if len(all_executions) + 3 >= budget else 0
                # print(r)
                return r
                return (len(all_executions)+1)/(budget+1)

        termination = ObstaclePlacementTermination()


        res = minimize(problem, algorithm, termination, verbose=1)

        ### You should only return the test cases
        ### that are needed for evaluation (failing or challenging ones)
        return good_single_obstacle_tests \
                + [e.testCase for e in
                   sorted([e for e in all_executions if e.min_distance < 1.5], key=lambda e: e.result)]


if __name__ == "__main__":
    generator = MHSGenerator("case_studies/mission2.yaml")
    generator.generate(200)
