from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np
import math


# Supported impedance modes
IMPEDANCE_MODES = {"fixed",}

# TODO: Maybe better naming scheme to differentiate between input / output min / max and pos/ori limits, etc.


class ImpedanceOperationalSpaceController(Controller):
    """
    It's based on the default osc controller from robosuite.
    We implemented impedance control on top of it.
    The theory based on the paper by Valency: https://doi.org/10.1115/1.1590685
    Also some code is adapted from shirkozlovsky's code.
    
    Addtional Args from OSC:
                    
                    impedance_stiffness: a tupe that contains the impedance model stifness coefficients.
                                        the length should be 6 and it corresponds to stiffness in the direction of (x, y, z, ax, ay, az)

                    impedance_damping: a matrix that contains the impedance model damping coefficients.
                                        the length should be 6 and it corresponds to damping in the direction of (x, y, z, ax, ay, az)

                    impedance_inertial: a matrix that contains the impedance model inertial coefficients.
                                        the length should be 6 and it corresponds to inertial in the direction of (x, y, z, ax, ay, az)
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 impedance_stiffness,
                 impedance_damping,
                 impedance_inertial,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 kp=150,
                 damping_ratio=1,
                 impedance_mode="fixed",
                 kp_limits=(0, 300),
                 damping_ratio_limits=(0, 100),
                 policy_freq=20,
                 position_limits=None,
                 orientation_limits=None,
                 interpolator_pos=None,
                 interpolator_ori=None,
                 control_ori=True,
                 control_delta=True,
                 uncouple_pos_ori=True,
                 **kwargs # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )
        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        # Control dimension
        self.control_dim = 6 if self.use_ori else 3
        self.name_suffix = "POSE" if self.use_ori else "POSITION"

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, "Error: Tried to instantiate IMP controller for unsupported " \
                                                  "impedance mode! Inputted impedance mode: {}, Supported modes: {}". \
            format(impedance_mode, IMPEDANCE_MODES)

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        # if self.impedance_mode == "variable":
        #     self.control_dim += 12
        # elif self.impedance_mode == "variable_kp":
        #     self.control_dim += 6

        # limits
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq
        self.dt = 1 / self.control_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals based on initial pos / ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)

        self.relative_ori = np.zeros(3)
        self.ori_ref = None

        # impedance coefficients
        self.impK = np.diag(np.array(impedance_stiffness))

        self.impB = np.diag(np.array(impedance_damping))
    
        self.impM = np.diag(np.array(impedance_inertial))

        # impedance model position
        self.x0     = 0
        self.x0_d   = 0
        self.x0_dd   = 0

        # external force acting on the eef
        self.F_int  = 0

        # Previous Jacobian for Jd calculation
        self.J_previous = np.zeros((6,6))

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action.
        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary
        if self.impedance_mode == "fixed":
            delta = action

        # If we're using deltas, interpret actions as such
        if self.use_delta:
            if delta is not None:
                scaled_delta = self.scale_action(delta)
                if not self.use_ori and set_ori is None:
                    # Set default control for ori since user isn't actively controlling ori
                    set_ori = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            else:
                scaled_delta = []
        # Else, interpret actions as absolute values
        else:
            if set_pos is None:
                set_pos = delta[:3]
            # Set default control for ori if we're only using position control
            if set_ori is None:
                set_ori = T.quat2mat(T.axisangle2quat(delta[3:6])) if self.use_ori else \
                    np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
            # No scaling of values since these are absolute values
            scaled_delta = delta

        # We only want to update goal orientation if there is a valid delta ori value OR if we're using absolute ori
        # use math.isclose instead of numpy because numpy is slow
        bools = [0. if math.isclose(elem, 0.) else 1. for elem in scaled_delta[3:]]
        if sum(bools) > 0. or set_ori is not None:
            self.goal_ori = set_goal_orientation(scaled_delta[3:],
                                                 self.ee_ori_mat,
                                                 orientation_limit=self.orientation_limits,
                                                 set_ori=set_ori)
        self.goal_pos = set_goal_position(scaled_delta[:3],
                                          self.ee_pos,
                                          position_limit=self.position_limits,
                                          set_pos=set_pos)

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0
        
        self.F_int = action[-3:]

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        
        It's based on the default osc controller from robosuite.
        We implemented impedance control on top of it.
        The theory based on the paper by Valency: https://doi.org/10.1115/1.1590685
    
        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        # update impedance model and x0, x0_d
        self._update_impedance_model()

        # calculate jacobian derivative
        Jd_full = (self.J_full - self.J_previous) / self.dt

        # calculate torque
        M_inv           = np.linalg.inv(self.impM)
        J_inv           = np.linalg.pinv(self.J_pos) 
        Jd_times_qd     = Jd_full @ self.joint_vel
        x_dd            = M_inv @ ( self.impK @ (self.x0-self.ee_pos) + self.impB @ (self.x0_d-self.ee_pos_vel) - self.F_int)
        self.torques    = self.mass_matrix @ J_inv @ (x_dd - Jd_times_qd[0:3])
        self.torques    += self.torque_compensation
        self.torques    -= self.J_pos.T @ self.F_int

        self.J_previous = self.J_full
        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def _update_impedance_model(self):
        """
        Impedance Eq: F_int-F0=K(x0-xr)+B(x0_d-xr_d)-Mxm_dd
        Solving the impedance equation and update x0, x0_d (virtual impedance model)
                                                  xr = real value
        """
        K, B, M     = self.impK, self.impB, self.impM
        M_inv       = np.linalg.inv(M)
        self.x0_dd  =  M_inv @ ( K @ (self.x0-self.ee_pos) + B @ (self.x0_d-self.ee_pos_vel) - self.F_int )
        self.x0_d   = self.x0_dd * self.dt
        self.x0     = self.x0_d * self.dt

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return 'Impedance_OSC' + self.name_suffix
