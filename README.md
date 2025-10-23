# rosmimic

Adaptation of [robomimic](https://github.com/ARISE-Initiative/robomimic) for creating datasets and deploying policies with ROS, tested for Diffusion Policy using the Franka robot in Gazebo simulation and in the real world.

## Docker

Allow access to local displays:

`xhost +local:docker`

Build with:
`docker build -t rosmimic .`

Open rosmimic folder in VSCode. Reopen and rebuild in devcontainer (Ctrl + Shift + P and search for those commands).

## Policy training pipeline example

Example of training a pick and insert gear diffusion policy using demos collected with a space mouse. The used gym environment is SERL, which maps to `robomimic/envs/env_franka_serl.py`. It is ROS based and uses two cameras and the SERL cartesian impedance controller. 

### Demo collection

Collect 10 demos with:

```
python robomimic/scripts/collect_demos.py --env SERL --n_rollouts 10 --dataset_path datasets/pick_insert_gear1.h5 --dataset_obs --video_path videos/pick_insert_gear1.mp4
```

Then more 10 demos:

```
python robomimic/scripts/collect_demos.py --env SERL --n_rollouts 10 --dataset_path datasets/pick_insert_gear2.h5 --dataset_obs --video_path videos/pick_insert_gear2.mp4
```

And so on... I do it five times to get 50 demos.  You can also change n_rollouts to collect them all at once but if something crashes in the meantime you have to restart from scratch.

### Model training

First setup your Weights and Biases account link with:

```
python robomimic/scripts/setup_macros.py
```

Then, in `robomimic/macros_private.py`, fill in `WANDB_API_KEY` with your wandb api key, and `WANDB_ENTITY` with your wandb username+team (e.g. vvkowalski-tu-wien).

Then train the diffusion policy using your collected demos with:

```
python robomimic/scripts/train.py --config robomimic/exps/templates/diffusion_policy.json --datasets pick_insert_gear1.h5 pick_insert_gear2.h5
```

The current config saves checkpoints every 25 epochs. 100 epochs of training takes 30 min in my office PC, and the resulting policy is OK-ish.

### Policy evaluation

Evaluate the performance of your training policy on the robot with: 

```
python robomimic/scripts/run_trained_agent.py --agent diffusion_policy_trained_models/pick_place_gear_50/20251023125651/models/model_epoch_100.pth --n_rollouts 10 --horizon 100 --seed 0
```

The arg. --horizon 100 means timeout after 100 steps (= 10 seconds at 10Hz). 

## Deployment examples

Results from Diffusion policies trained from 50 demos for ~30 minutes each.

### Gazebo environment, simple pick

Policy trained from scripted demos (**bad grasp because the gripper was being closed too early in the collected demos**):

https://github.com/user-attachments/assets/a60c484d-b6a3-425a-aaf3-e8ce03b52dc6

### Real environment, simple pick (with Daniel's cartesian impedance damping ratio controller)

Policy trained from teleoperated demos (with SpaceMouse):

https://github.com/user-attachments/assets/37e1e820-3c57-417b-a44d-01e707372b20

### Real environment, contact-rich (with SERL cartesian impedance controller)

Policy trained from teleoperated demos (with SpaceMouse):

https://github.com/user-attachments/assets/37e1e820-3c57-417b-a44d-01e707372b20


