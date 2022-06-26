"""
This code contains examples of how to call and use the SCoPP-Monitoring module.
"""

#Testing Git to see if this uploads to the online repository

# Import the necessary modules:
#from Swarm_Surveillance.SCoPP 
import monitoring_algorithms
#from Swarm_Surveillance.SCoPP 
import environments as envs

# Initialize environment class
# environment = envs.MediumLafayetteFLood(0)
# environment = envs.Baseline_Envirnonment("baseline")
# environment = envs.ShastaBuffaloSmall()


# Initialize monitoring algorithm instance
way_point_allocator = monitoring_algorithms.QLB(environment, 10, priority_points=10 ,plot="full")

# Run the algorithm on the given environment and display all information
way_point_allocator.run(info="verbose")