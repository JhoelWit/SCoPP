"""
This code contains examples of how to call and use the SCoPP-Monitoring module.
"""

#Testing Git to see if this uploads to the online repository

# Import the necessary modules:
#from Swarm_Surveillance.SCoPP 
import monitoring_algorithms
#from Swarm_Surveillance.SCoPP 
import environments as envs
import SCoPP_settings as settings

# Initialize environment class
# environment = envs.SmallLafayetteFLood(0) # Use 5 robots
environment = envs.MediumLafayetteFLood(0) # Use 20 robots
# environment = envs.LargeLafayetteFLood(0) # Use 75 robots
# environment = envs.Baseline_Envirnonment("baseline") # Use 3 robots
# environment = envs.ShastaBuffaloSmall()

test_settings = settings.algorithm(tests=[None])
# test_settings = settings.algorithm(tests=[None])
# Initialize monitoring algorithm instance
way_point_allocator = monitoring_algorithms.QLB(environment, 20, priority_points=10 ,plot="full", algorithm_settings=test_settings)

# Run the algorithm on the given environment and display all information
way_point_allocator.run(info="verbose")