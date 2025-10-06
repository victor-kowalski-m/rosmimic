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
