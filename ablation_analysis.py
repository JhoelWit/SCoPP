"""
This is an ablation study for the partitioning step of SCoPP
"""

# Import the necessary modules:
import monitoring_algorithms
import environments as envs
import SCoPP_settings as settings
import copy

import csv

# Initialize environment class
# environment = envs.MediumLafayetteFLood(0)

# Initialize monitoring algorithm instance
#way_point_allocator = monitoring_algorithms.QLB(environment, 10, plot="false")

# Initialize bias factor for parametric testing
settings_  = settings.algorithm()

# Running the algorithm a set number of times, and recording metrics with and without ablation. 
test_runs = 10

# Repeatable function to run different environments and settings, with writing to the CSV file afterwards.
def run_tests(num_tests: int = 10, way_point_allocator: monitoring_algorithms.QLB = None, algo: str = None, test: str = "ablation", environment: str = "Medium", writer: csv.writer = None):
    for _ in range(num_tests):
            temp_wpa = copy.deepcopy(way_point_allocator)
            info_dict = temp_wpa.run(info="false")

            mission_time = info_dict["mission"]["completion_time"]
            part_time = info_dict["time"]["initial_partitioning"]
            path_plan_time = info_dict["time"]["path_planning"]
            algo_time = info_dict["time"]["total_comp"]

            # Writing the information
            writer.writerow([environment, algo, test, mission_time, part_time, path_plan_time, algo_time])

# We'll be writing the data into a CSV file
with open('ablation_study_variable_robots.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Map', 'Algorithm', 'Ablation Mode', 'Mission Time', 'Partition Time', 'Path Planning Time', 'Algorithm Time'])

    # With ablation  - Small map
    environment = envs.SmallLafayetteFLood(0)
    test_settings = settings.algorithm(tests=["ablate-partition"])
    robots = 5
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    algo = 'SCoPP'
    test = 'Partition Ablation'
    map = 'Small'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # Without ablation - Small map
    test_settings = settings.algorithm(tests=[None])
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    test = 'No Ablation'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # With ablation  - Medium map
    environment = envs.MediumLafayetteFLood(0)
    test_settings = settings.algorithm(tests=["ablate-partition"])
    robots = 20
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    algo = 'SCoPP'
    test = 'Partition Ablation'
    map = 'Medium'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # Without ablation - Medium map
    test_settings = settings.algorithm(tests=[None])
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    test = 'No Ablation'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # With ablation  - Large map
    environment = envs.VeryLargeLafayetteFLood(0)
    test_settings = settings.algorithm(tests=["ablate-partition"])
    robots = 75
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    algo = 'SCoPP'
    test = 'Partition Ablation'
    map = 'Large'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # Without ablation - Large map
    test_settings = settings.algorithm(tests=[None])
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    test = 'No Ablation'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # With ablation  - Baseline map
    environment = envs.Baseline_Envirnonment("baseline")
    test_settings = settings.algorithm(tests=["ablate-partition"])
    robots = 3
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    algo = 'SCoPP'
    test = 'Partition Ablation'
    map = 'Baseline'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

    # Without ablation - Baseline map
    test_settings = settings.algorithm(tests=[None])
    way_point_allocator = monitoring_algorithms.QLB(environment, robots, plot="false", algorithm_settings=test_settings)
    test = 'No Ablation'
    run_tests(way_point_allocator=way_point_allocator, algo=algo, test=test, environment=map, writer=writer)

