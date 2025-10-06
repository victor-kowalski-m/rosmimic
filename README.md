# rosmimic

Adaptation of [robomimic](https://github.com/ARISE-Initiative/robomimic) for creating datasets and deploying policies with ROS, tested for Diffusion Policy using the Franka robot in Gazebo simulation and in the real world.

## Docker

To build, run:
`docker build -t rosmimic .`

To run without GPU (CPU only), run:
`docker run -it rosmimic`

To run with GPU (if available), run:
`docker run --gpus all -it rosmimic`

## Policy training

Collect demos (scripted or teleoperated via SpaceMouse) with:
`python scripts/collect_demos.py` 

Train with:
`python scripts/train.py`

Deploy with:
`python scripts/run_trained_agent.py`

(Use the appropriate command-line arguments for each.)

## Deployment examples

Diffusion policies for a pick task were evaluated in Gazebo simulation and in the real world with the Franka robot. Both policies were trained from 50 demos for ~30 minutes.

### Gazebo environment

Policy trained from scripted demos (**bad grasp because the gripper was being closed too early in the collected demos**):

https://github.com/user-attachments/assets/a60c484d-b6a3-425a-aaf3-e8ce03b52dc6

### Real environment 

Policy trained from teleoperated demos (with SpaceMouse):

https://github.com/user-attachments/assets/37e1e820-3c57-417b-a44d-01e707372b20

