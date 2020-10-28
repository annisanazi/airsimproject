# Airsim Project for Collision Avoidance Systtem
This is a project about simulating an automatic obstacle avoidance system in an autonomous car using reinforcement learning. The simulator used is Airsim. Open source simulator for autonomous vehicles built on Unreal Engine / Unity, from Microsoft AI & Research. I use coding from [Airsim Github](https://github.com/microsoft/AirSim).

Most road accidents are caused by careless drivers and from these accidents result in high mortality rates. Autonomous car technology with a collision avoidance system is expected to reduce the death rate. Because this car has a number of sensors that continue to be active, monitoring the area around the car to prevent accidents. The developed system allows the car to avoid physical contact with surrounding objects.
Collision avoidance system by using a camera requires a platform that is able to make a simulation environment that is close to the real situation. The simulator used in this study is Microsoft AirSim. Microsoft AirSim can be controlled by programming based on the reinforcement learning method, but the use of the reinforcement learning method requires considerable training data to learn a number of conditions to be able to take appropriate actions, so in this study using a simulator to run an autonomous car, because the simulator has several advantages which is more efficient and safer.
In this study the method of reinforcement learning is used with the Deep Q-Network algorithm to replace the look-up table in Q-learning. The test was carried out several times with several different provisions, namely the number of obstacles and the threshold distance value. This study produces graphs of rewards per episode that are increasingly converging and graphs of loss functions that are increasingly declining.In this study conducted several tests with several different provisions, namely the number of obstacles and the threshold distance value. This study produces graphs of rewards per episode that are increasingly converging and graphs of loss functions that are increasingly declining.


### Environment
For the environment, I build my own environment using Unreal Engine. The name of the envinroment is Simple Road. This is the preview of the environment
![Simple Road Environement]()

You can download the environment from [here](https://drive.google.com/drive/folders/1H1uw0oUb90fkZdt7Hqt84SoDFrxWOGP5?usp=sharing)
