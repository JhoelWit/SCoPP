"""
This code contains examples of how to call and use the SCoPP-Monitoring module.
"""

# Import the necessary modules:
#from Swarm_Surveillance.SCoPP 
import monitoring_algorithms
#from Swarm_Surveillance.SCoPP 
import environments as envs

import SCoPP_settings as settings

# Initialize environment class
environment = envs.MediumLafayetteFLood(0)

# Initialize monitoring algorithm instance
#way_point_allocator = monitoring_algorithms.QLB(environment, 10, plot="false")

# Initialize bias factor for parametric testing
parametric_settings  = settings.algorithm()
# Run the algorithm on the given environment and display all information


j = 2
while j < 3:
    comptotal = 0
    misstotal = 0
    i = 0
    if j == 0:
        bias = 0
        parametric_settings.bias_factor = bias
        way_point_allocator = monitoring_algorithms.QLB(environment, 20, plot="false",algorithm_settings=parametric_settings)
        x = 20
    elif j == 1:
        bias = 0.2
        parametric_settings.bias_factor = bias
        way_point_allocator = monitoring_algorithms.QLB(environment, 20, plot="false",algorithm_settings=parametric_settings)
        x = 20
    elif j == 2:
        bias = 0.4
        parametric_settings.bias_factor = bias
        way_point_allocator = monitoring_algorithms.QLB(environment, 20, plot="false",algorithm_settings=parametric_settings)
        x = 20
    elif j == 3:
        bias = 0.6
        parametric_settings.bias_factor = bias
        way_point_allocator = monitoring_algorithms.QLB(environment, 20, plot="false",algorithm_settings=parametric_settings)
        x = 20
    elif j == 4:
        bias = 0.8
        parametric_settings.bias_factor = bias
        way_point_allocator = monitoring_algorithms.QLB(environment, 20, plot="false",algorithm_settings=parametric_settings)
        x = 20
    elif j == 5:
        bias = 1
        parametric_settings.bias_factor = bias
        way_point_allocator = monitoring_algorithms.QLB(environment, 20, plot="false",algorithm_settings=parametric_settings)
        x = 20
    elif j == 6:
        way_point_allocator = monitoring_algorithms.QLB(environment, 75, plot="false")
        x = 75
    elif j == 7:
        way_point_allocator = monitoring_algorithms.QLB(environment, 100, plot="false")
        x = 100
    elif j == 8:
        way_point_allocator = monitoring_algorithms.QLB(environment, 125, plot="false")
        x = 125
    elif j == 9:
        way_point_allocator = monitoring_algorithms.QLB(environment, 150, plot="false")
        x = 150
    for i in range(10):
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        paths, prioritypaths, comptime, misstime = way_point_allocator.run(info="false")
        comptotal+=comptime
        misstotal+=misstime
        i+=1
        # print("case ", (j+1), " run ", i)
        # print("comp time is ", comptime, "mission time is ", misstime)
    compaverage = comptotal / 10
    missaverage = misstotal / 10
    j+=1
    print("case ", j)
    print('bias is ',parametric_settings.bias_factor)
    print("total comp time is ", comptotal, "total mission time is ", misstotal)
    print("average computation time over 10 runs is ", compaverage, " average mission time over 10 runs is ", missaverage)
    