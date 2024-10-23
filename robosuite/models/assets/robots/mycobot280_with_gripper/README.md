# mujoco_mycobot
This package is for mjcf of mycobot with gripper.
The current version of the mjcf is not accurate and has been written arbitrarily.
Adjustments to the parameters are required for the real2sim transition.

## How to use

``` bash
python3 -m mujoco.viewer -mjcf mycobot_with_gripper.xml
```

![mycobot_with_gripper](https://github.com/soonhyo/mujoco_mycobot/blob/main/image.png)

## License
Licensed under standard three-clause BSD license (same as ROS Core), 
Copyright 2020-2023 Elephant Robotics. [Copy of the license](LICENSE).

official mycobot_ros repository link: https://github.com/elephantrobotics/mycobot_ros/tree/ros_gripper
