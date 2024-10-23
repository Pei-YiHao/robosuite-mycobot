import robosuite as suite
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import choose_robots
import PIL.Image as Image
import requests
from typing import Any, Optional
import json
from base64 import b64decode, b64encode
from numpy.lib.format import descr_to_dtype, dtype_to_descr
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def default(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return {
            "__numpy__": b64encode(obj.data if obj.flags.c_contiguous else obj.tobytes()).decode("ascii"),
            "dtype": dtype_to_descr(obj.dtype),
            "shape": obj.shape,
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def object_hook(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        shape = dct["shape"]
        return np_obj.reshape(shape) if shape else np_obj[0]  # Scalar test
    return dct


_dumps = json.dumps
_loads = json.loads
_dump = json.dump
_load = json.load


def dumps(*args, **kwargs):
    kwargs.setdefault("default", default)
    return _dumps(*args, **kwargs)


def loads(*args, **kwargs):
    kwargs.setdefault("object_hook", object_hook)
    return _loads(*args, **kwargs)


def dump(*args, **kwargs):
    "test"
    kwargs.setdefault("default", default)
    return _dump(*args, **kwargs)


def load(*args, **kwargs):
    kwargs.setdefault("object_hook", object_hook)
    return _load(*args, **kwargs)


def patch():
    """Monkey patches the json module in order to support serialization/deserialization of Numpy arrays and scalars."""
    json.dumps = dumps
    json.loads = loads
    json.dump = dump
    json.load = load

patch()

class robot_control:
    def __init__(self, env_name="Lift", robot_name="UR5e", controller_name="OSC_POSE"):
        """
        初始化环境、机械臂、控制器配置
        """
        # 配置参数
        self.env_name = env_name
        self.robot_name = robot_name
        self.controller_name = controller_name

        # 机械臂关节维度
        self.joint_dim = 7  # IIWA机器人一般有7个自由度

        # 控制器配置
        self.controller_config = suite.load_controller_config(default_controller=self.controller_name)

        # 环境初始化
        self.env = suite.make(
            env_name=self.env_name,
            robots=self.robot_name,
            controller_configs=self.controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,  # 我们手动获取摄像机图像
            ignore_done=True,
            control_freq=20,
        )
        #self.set_camera()
        self.env.reset()

    def set_camera(self):
        self.env.sim.model.cam_pos[self.env.sim.model.camera_name2id("agentview")] = [-0.5162175144302095, 0.22389066505019817, 1.33095515612768]
        self.env.sim.model.cam_quat[self.env.sim.model.camera_name2id("agentview")] = [0.6043705344200134, 0.16804856061935425, -0.2970672845840454, -0.7198938131332397]
        self.env.render()
        
    def step(self, action):
        """
        接收七维动作，执行机械臂的相应动作：
        action 是一个7维向量 [x, y, z, rx, ry, rz, gripper_value]
        """
        assert len(action) == 7, "动作输入必须是7维的：x, y, z, rx, ry, rz, gripper_value"
        
        # 解析传入的动作
        ee_pos = action[:3]  # [x, y, z]
        ee_ori = action[3:6]  # [rx, ry, rz]
        gripper_value = action[6]  # 夹爪的动作值

        # 定义动作：机械臂控制部分 + 夹爪控制部分
        arm_action = np.concatenate([ee_pos, ee_ori])  # 6维的机械臂运动
        total_action = np.concatenate([arm_action, [gripper_value]])  # 包含夹爪的7维动作
        print("执行动作：", total_action)
        # 执行动作
        obs, reawrd, done, info = self.env.step(total_action)
        print("执行结果：", reawrd, done)
        #self.env.render()

    def get_image(self):
        """
        从摄像机获取当前帧图像
        """
        #self.set_camera()
        img = self.env.sim.render(
            width=256, height=256, camera_name="agentview"
        )
        return img
    
    def get_ee_position(self):
        """
        获取机械臂末端执行器的位置
        """
        ee_position = self.env.robots[0]._hand_pos
        return ee_position
    
    def get_ee_pose(self):
        """
        获取机械臂末端执行器的七维位姿向量
        :return: 七维向量 [x, y, z, rx, ry, rz, gripper_value]
        """
        # 获取末端执行器的4x4位姿矩阵
        ee_pose_matrix = self.env.robots[0]._hand_pose

        # 提取位置部分
        position = ee_pose_matrix[:3, 3]  # x, y, z

        # 提取旋转矩阵并转换为欧拉角
        rotation_matrix = ee_pose_matrix[:3, :3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')  # 转换为欧拉角 rx, ry, rz

        # 获取夹爪状态 (根据需要设置)
        gripper_value = self.env.robots[0].gripper.current_action if hasattr(self.env.robots[0], 'gripper') else 0
        ee_pose = np.concatenate([position, euler_angles])

        return ee_pose, gripper_value
    
    def render_from_camera(self, camera_name, width=256, height=256):
        """
        渲染指定摄像机视角的图像并对其进行旋转180度和左右镜像变换
        :param camera_name: 摄像机的名字（例如 'agentview' 或 'frontview'）
        :param width: 图像宽度
        :param height: 图像高度
        :return: 处理后的图像
        """
        # 渲染图像
        img = self.env.sim.render(
            width=width,
            height=height,
            camera_name=camera_name
        )

        # 将 NumPy 数组转换为 PIL 图像
        img = Image.fromarray(img)
        
        # 旋转180度
        img = img.rotate(180)
        
        # 左右镜像变换
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 转回 NumPy 数组
        img = np.array(img)

        return img

    
    def render_both_views(self):
        """
        渲染两个视角的离屏图像
        """
        # 渲染 'agentview' 视角
        agent_view_img = self.render_from_camera("agentview")

        # 渲染 'frontview' 视角
        front_view_img = self.render_from_camera("frontview")

        # 显示两幅图像
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(agent_view_img)
        axes[0].set_title("Agent View")
        axes[0].axis('off')

        axes[1].imshow(front_view_img)
        axes[1].set_title("Front View")
        axes[1].axis('off')

        plt.show()

class robot_control_server(robot_control):
    def __init__(self, ip="127.0.0.1", port=7999, env_name="Stack", robot_name="Panda", controller_name="OSC_POSE"):
        """
        初始化环境、机械臂、控制器配置
        """
        self.ip = ip
        self.port = port
        self.unnorm_key = "stack_d4_ep2500"
        self.instruction = "Stack the green block on the red one"
        super().__init__(env_name, robot_name, controller_name)

    def _query_for_action(self, img, instruction):
    # Convert the NumPy array to a list
        
        # Prepare the data to send in the POST request
        data = {
            'instruction': instruction,
            'image': img,
            "unnorm_key":self.unnorm_key
        }
        # Make the POST request
        retries = 5
        while retries > 0:
            try:
                res = requests.post(
                    url=f"http://{self.ip}:{self.port}/act",  # Replace with your actual URL
                    json=data
                )
                break
            except:
                retries -= 1
                continue
        
        return res.json()
    
    def get_image(self):
        img = super().get_image()
        img = Image.fromarray(img).rotate(180)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
    def step(self):
        print("当前位姿", self.get_ee_pose())
        
        # 获取图像并反转
        img = self.get_image()
        #self.render_both_views()
        # 指令
        instruction = self.instruction
        
        # 获取动作
        action = self._query_for_action(np.array(img), instruction).copy()
        
        # 缩放动作
        action[:3] *= [272., 294., 172.]  # [886.7, 230.1, 827.2]
        action[3:6] *= [np.pi/2, np.pi/2, np.pi/2]  
        action[-1:] = 1-action[-1:]*2  # 控制夹爪
        # 执行动作
        super().step(action)
        
        # 每次执行完动作后渲染
        #self.env.render()

        
        
# 使用示例
if __name__ == "__main__":
    # 创建robot_control对象，默认配置
    robot = robot_control_server()

    while True:
        robot.step()
