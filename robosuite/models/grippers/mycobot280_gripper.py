"""
6-DoF gripper with its open/close variant
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Mycobot280GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/mycobot280_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.,0.,0.,0.,0.,0.]).astype(np.float64)

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                'gripper_left3_collision',
                'gripper_left2_collision',
                'gripper_left1_collision',
                'left_finger_tip_collision'
            ],
            "right_finger": [
                'gripper_right3_collision',
                'gripper_right2_collision',
                'gripper_right1_collision',
                'right_finger_tip_collision'
            ],
            "left_fingerpad": ['left_finger_tip_collision'],
            "right_fingerpad": ['right_finger_tip_collision'],
        }


class Mycobot280Gripper(Mycobot280GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
