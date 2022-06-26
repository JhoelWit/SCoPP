"""
This code contains examples of how to call and use the SCoPP-Monitoring module.
"""

# Import the necessary modules:
#from Swarm_Surveillance.SCoPP 
import monitoring_algorithms
#from Swarm_Surveillance.SCoPP 
import environments as envs
import numpy as np

# Initialize environment class
environment = envs.Baseline_Envirnonment("baseline")
# environment = envs.MediumLafayetteFLood(0)
# environment = envs.LargeLafayetteFLood(0)


# Initialize monitoring algorithm instance
#way_point_allocator = monitoring_algorithms.QLB(environment, 10, plot="false")

# Run the algorithm on the given environment and display all information

missionsd = []
j = 0
while j < 1:
    comptotal = 0
    misstotal = 0
    ppsurveytotal = 0
    i = 0
    if j == 0:
        x = 10
        ppoints = 100
        way_point_allocator = monitoring_algorithms.QLB(environment, x, priority_points=ppoints, plot="false")
        
    elif j == 1:
        x = 10
        ppoints = 10
        way_point_allocator = monitoring_algorithms.QLB(environment, x, priority_points=ppoints, plot="false")
        
    elif j == 2:
        x = 10
        ppoints = 100
        way_point_allocator = monitoring_algorithms.QLB(environment, x, priority_points=ppoints, plot="false")
        
    elif j == 3:
        x = 10
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    elif j == 4:
        x = 40
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    elif j == 5:
        x = 50
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    elif j == 6:
        x = 75
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    elif j == 7:
        x = 100
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    elif j == 8:
        x = 125
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    elif j == 9:
        x = 150
        way_point_allocator = monitoring_algorithms.QLB(environment, x, plot="false")
        
    runs = 10
    for i in range(runs):
        way_point_allocator = monitoring_algorithms.QLB(environment, x, priority_points=ppoints, plot="false")
        paths, prioritypaths, comptime, misstime, ppsurvey = way_point_allocator.run(info="false")
        missionsd.append(misstime)
        comptotal+=comptime
        misstotal+=misstime
        ppsurveytotal += ppsurvey
        i+=1
        print("case ", (j+1), " run ", i)
        # print("comp time is ", comptime, "mission time is ", misstime)
    compaverage = comptotal / runs
    missaverage = misstotal / runs
    ppsurveyaverage = ppsurveytotal / runs
    j+=1
    print("case ", j)
    # print("total comp time is ", comptotal, "total mission time is ", misstotal)
    print("average computation time over", runs, "runs is ", compaverage, " average mission time over", runs," runs is ", missaverage,'average ppoint survey time is',ppsurveyaverage)
    print('priority points for run',j,':',x*ppoints)
    # standarddev = 0
    # for i in range(runs):
    #     standarddev += 1/runs * (missionsd[i] - missaverage)**2
    # print('Standard Deviation is ',np.sqrt(standarddev))
