"""
This code contains examples of how to call and use the SCoPP-Monitoring module.
"""


# Import the necessary modules:
#from Swarm_Surveillance.SCoPP 
import monitoring_algorithms
#from Swarm_Surveillance.SCoPP 
import environments as envs

# Initialize environment class
environment = envs.LargeLafayetteFLood(0)

# Initialize monitoring algorithm instance
#way_point_allocator = monitoring_algorithms.QLB(environment, 10, plot="false")

# Run the algorithm on the given environment and display all information


j = 0
while j != 1:
    comptotal = 0
    misstotal = 0
    i = 0
    if j == 0:
        way_point_allocator = monitoring_algorithms.QLB(environment, 10, plot="false")
        x = 10
    elif j == 1:
        way_point_allocator = monitoring_algorithms.QLB(environment, 15, plot="false")
        x = 15
    elif j == 2:
        way_point_allocator = monitoring_algorithms.QLB(environment, 25, plot="false")
        x = 25
    elif j == 3:
        way_point_allocator = monitoring_algorithms.QLB(environment, 30, plot="false")
        x = 30
    elif j == 4:
        way_point_allocator = monitoring_algorithms.QLB(environment, 40, plot="false")
        x = 40
    elif j == 5:
        way_point_allocator = monitoring_algorithms.QLB(environment, 50, plot="false")
        x = 50
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
        print("case ", (j+1), " run ", i)
        print("comp time is ", comptime, "mission time is ", misstime)
    compaverage = comptotal / 10
    missaverage = misstotal / 10
    j+=1
    print("case ", j)
    print("total comp time is ", comptotal, "total mission time is ", misstotal)
    print("average computation time over 10 runs is ", compaverage, " average mission time over 10 runs is ", missaverage)
    
