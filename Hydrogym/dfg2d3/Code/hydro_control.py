"""
hydro_control.py

HydroGym wrapper for the existing Firedrake DFG 2D-3 solver in fire1.py.

Expected folder layout
----------------------
project/
    fire1.py
    hydro_control.py

fire1.py must provide:
    fem_step(control)
    reset_state()
    get_observation()
    set_control(control)
    calculate_coefficients()
    pressure_difference()
    up, u_out, p_out, u_old, mesh, bcs
    dt_value, time_value, omega

The CFD/FEM solve remains inside fire1.py.
HydroGym only wraps it as an RL environment.
"""

import argparse
import os
import numpy as np

# Firedrake recommends one OpenMP thread per MPI process.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import fire1 as fem

from hydrogym import FlowEnv
from hydrogym.core import PDEBase, TransientSolver


# ============================================================
# 1. HydroGym PDE adapter
# ============================================================

class DFGFlow(PDEBase):
    """
    Thin HydroGym wrapper around the existing Firedrake state in fire1.py.

    This class DOES NOT define a new Navier-Stokes weak form.
    It only tells HydroGym:
        - how many actions exist
        - how many observations exist
        - how to reset/read the Firedrake state
        - how to evaluate the control objective
    """

    def __init__(
        self,
        max_omega=10.0,
        lift_weight=0.2,
        control_weight=1.0e-3,
        **kwargs,
    ):
        # HydroGym uses this to construct the action range.
        self.MAX_CONTROL = float(max_omega)

        self.lift_weight = float(lift_weight)
        self.control_weight = float(control_weight)

        self.t = float(fem.time_value)
        self.last_result = None

        # Keep references to the actual Firedrake objects.
        self.mesh = fem.mesh
        self.q = fem.up

        # Initial state for compatibility / inspection.
        self.q0 = fem.up.copy(deepcopy=True)

    # --------------------------------------------------------
    # HydroGym interface
    # --------------------------------------------------------

    @property
    def num_inputs(self):
        # One actuator: cylinder angular velocity omega
        return 1

    @property
    def num_outputs(self):
        # Observation = [Cd, Cl, pressure difference]
        return 3

    def load_mesh(self, name=None):
        # Mesh is already created in fire1.py.
        return fem.mesh

    def initialize_state(self):
        # State is already created in fire1.py.
        self.q = fem.up
        return self.q

    def init_bcs(self):
        # Boundary conditions are already created in fire1.py.
        return fem.bcs

    def collect_bcs(self):
        return fem.bcs

    @property
    def state(self):
        return fem.up

    def set_state(self, q):
        fem.up.assign(q)

        # Keep Crank-Nicolson history consistent.
        fem.u_old.assign(fem.u_out)

        self.q = fem.up

    def copy_state(self, deepcopy=True):
        return fem.up.copy(deepcopy=deepcopy)

    def reset(self, q0=None, t=0.0):
        """
        Reset one RL episode.

        Current version restarts from the original DFG initial state:
            u = 0
            p = 0
            omega = 0
            t = 0
        """
        fem.reset_state()

        if q0 is not None:
            self.set_state(q0)

        self.t = float(fem.time_value)
        self.last_result = None
        self.q = fem.up

        return self.q

    def set_control(self, act=None):
        """
        Directly update cylinder angular velocity.
        """
        if act is None:
            value = 0.0
        else:
            arr = np.asarray(act, dtype=float).reshape(-1)
            value = float(arr[0])

        value = np.clip(
            value,
            -self.MAX_CONTROL,
            self.MAX_CONTROL,
        )

        fem.set_control(value)

        return np.array([value], dtype=np.float32)

    def advance_time(self, dt, act=None):
        """
        Included for HydroGym API compatibility.

        The actual time advancement is done by fem.fem_step()
        inside DFGFiredrakeSolver.step().
        """
        return self.set_control(act)

    def get_observations(self):
        """
        Current first-pass observation:
            [Cd, Cl, dp]

        Later this should preferably be extended with wake velocity probes.
        """
        obs = fem.get_observation()
        return np.asarray(obs, dtype=np.float32)

    def evaluate_objective(self, q=None):
        """
        Objective J to MINIMIZE.

        HydroGym interprets evaluate_objective() as the cost,
        and converts it to reward internally.

        J = Cd + alpha*Cl^2 + beta*(omega/max_omega)^2
        """
        CD, CL = fem.calculate_coefficients()

        omega_value = float(fem.omega)

        normalized_control = omega_value / self.MAX_CONTROL

        J = (
            CD
            + self.lift_weight * CL**2
            + self.control_weight * normalized_control**2
        )

        return float(J)

    def render(self, **kwargs):
        # ParaView output can remain in Firedrake.
        return None


# ============================================================
# 2. HydroGym transient solver adapter
# ============================================================

class DFGFiredrakeSolver(TransientSolver):
    """
    HydroGym transient-solver interface.

    IMPORTANT:
    No HydroGym CFD discretization is used here.

    Every call to step() calls:
        fire1.fem_step(control)

    Therefore the actual CFD solver is still the user's
    Firedrake Crank-Nicolson Navier-Stokes code.
    """

    def __init__(self, flow, dt=fem.dt_value, **kwargs):
        self.flow = flow
        self.dt = float(dt)

        # Protect against accidentally running HydroGym with a
        # timestep different from the one hard-coded in fire1.py.
        if not np.isclose(self.dt, fem.dt_value):
            raise ValueError(
                "DFGFiredrakeSolver dt must equal fire1.dt_value. "
                f"HydroGym dt={self.dt}, fire1 dt={fem.dt_value}"
            )

    def step(self, iter, control=None, **kwargs):
        if control is None:
            control = np.array([0.0], dtype=float)

        # Convert HydroGym action -> physical omega and clip it.
        actual_control = self.flow.set_control(control)

        # ----------------------------------------------------
        # THIS IS THE ACTUAL FIREDRAKE FEM TIMESTEP
        # ----------------------------------------------------
        result = fem.fem_step(actual_control)

        self.flow.t = float(result["time"])
        self.flow.last_result = result
        self.flow.q = fem.up

        return self.flow

    def reset(self):
        fem.reset_state()

        self.flow.t = float(fem.time_value)
        self.flow.last_result = None
        self.flow.q = fem.up

        return self.flow


# ============================================================
# 3. Build HydroGym environment
# ============================================================

def make_env(
    max_omega=10.0,
    num_substeps=20,
    max_time=8.0,
):
    """
    One HydroGym action is held for num_substeps Firedrake timesteps.

    DFG:
        dt = 1/1600

    With num_substeps=20:
        action interval = 20/1600 = 0.0125
        t=0 -> 8 requires 640 RL environment steps.
    """

    action_dt = num_substeps * fem.dt_value
    max_steps = int(np.ceil(max_time / action_dt))

    env_config = {
        "flow": DFGFlow,

        "flow_config": {
            "max_omega": max_omega,
            "lift_weight": 0.2,
            "control_weight": 1.0e-3,
        },

        # Custom wrapper around YOUR Firedrake solver.
        "solver": DFGFiredrakeSolver,

        "solver_config": {
            "dt": fem.dt_value,
        },

        "actuation_config": {
            "num_substeps": num_substeps,
            "reward_aggregation": "mean",
        },

        "max_steps": max_steps,
    }

    return FlowEnv(env_config)


# ============================================================
# 4. Simple environment test
# ============================================================

def test_environment(num_steps=5):
    """
    Test HydroGym <-> Firedrake connection without training.
    """

    env = make_env()

    obs, info = env.reset()

    print("")
    print("===== HydroGym DFG test =====")
    print("action_space      =", env.action_space)
    print("observation_space =", env.observation_space)
    print("initial obs       =", obs)
    print("")

    for k in range(num_steps):
        # Start with random control only to verify the connection.
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"env step={k + 1:4d}  "
            f"t={fem.time_value:.6f}  "
            f"omega={float(fem.omega): .6f}  "
            f"Cd={obs[0]: .6f}  "
            f"Cl={obs[1]: .6f}  "
            f"dp={obs[2]: .6f}  "
            f"reward={reward: .6f}"
        )

        if terminated or truncated:
            break

    env.close()


# ============================================================
# 5. PPO training
# ============================================================

def train_ppo(
    total_timesteps=10_000,
    model_name="ppo_dfg_rotation",
):
    """
    PPO training using Stable-Baselines3.

    Start with a SMALL total_timesteps value.
    Each RL step contains many expensive Firedrake solves.
    """

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(make_env())

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./hydrogym_logs/",
        n_steps=256,
        batch_size=64,
    )

    model.learn(
        total_timesteps=total_timesteps
    )

    model.save(model_name)

    env.close()

    print("")
    print(f"saved model: {model_name}.zip")


# ============================================================
# 6. Command line
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train PPO instead of only testing the environment.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of environment steps for the connection test.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="PPO training timesteps.",
    )

    args = parser.parse_args()

    if args.train:
        train_ppo(
            total_timesteps=args.timesteps
        )
    else:
        test_environment(
            num_steps=args.steps
        )


if __name__ == "__main__":
    main()
