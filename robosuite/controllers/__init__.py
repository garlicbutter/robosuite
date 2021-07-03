from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .osc import OperationalSpaceController
from .imp_osc import ImpedanceOperationalSpaceController
from .joint_pos import JointPositionController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController


CONTROLLER_INFO = {
    "JOINT_VELOCITY":  "Joint Velocity",
    "JOINT_TORQUE":    "Joint Torque",
    "JOINT_POSITION":  "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE":     "Operational Space Control (Position + Orientation)",
    "IMP_POSITION":     "Impednace based control in operational space(Position Only)",
    "IMP_POSE":     "Impednace based control in operational space (Position + Orientation)",
    "IK_POSE":      "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
