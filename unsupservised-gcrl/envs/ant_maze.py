import os
import jax
import mujoco
import xml.etree.ElementTree as ET

from brax import base
from brax import math
from flax import struct
from brax.io import mjcf
from jax import numpy as jnp
from typing import Tuple, Dict
from brax.envs.base import PipelineEnv, State

@struct.dataclass
class EvalGoals:
	eval_goals: Dict[str, jax.Array]

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = 'r'
GOAL = G = 'g'

U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

U_MAZE_EASY_EVAL = [[1, 1, 1, 1, 1],
                    [1, R, G, G, 1],
                    [1, 1, 1, G, 1],
                    [1, G, G, G, 1],
                    [1, 1, 1, 1, 1]]

U_MAZE_HARD_EVAL = [[1, 1, 1, 1, 1],
                    [1, R, 0, 0, 1],
                    [1, 1, 1, 0, 1],
                    [1, G, 0, 0, 1],
                    [1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_EASY_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, G, 1, 1, G, G, 1],
                    [1, G, G, 1, G, G, G, 1],
                    [1, 1, G, G, G, 1, 1, 1],
                    [1, G, G, 1, G, G, G, 1],
                    [1, G, 1, G, G, 1, G, 1],
                    [1, G, G, G, 1, G, G, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_HARD_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 1, 1, G, G, 1],
                    [1, 0, 0, 1, 0, 0, G, 1],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, G, 1, G, G, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_EASY_EVAL =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, R, G, G, G, 1, G, G, G, G, G, 1],
                            [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                            [1, G, G, G, G, G, G, 1, G, G, G, 1],
                            [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                            [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                            [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                            [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_HARD_EVAL =     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 1, G, 0, 0, 1],
                                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 1, G, 1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5

def find_robot(structure, size_scaling):
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                return i * size_scaling, j * size_scaling
            
def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])

    return jnp.array(goals)

# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
        maze_eval_layouts = {'easy': U_MAZE_EASY_EVAL, 'hard': U_MAZE_HARD_EVAL}
    elif maze_layout_name == "u_maze_sg":
        maze_layout = U_MAZE_HARD_EVAL
        maze_eval_layouts = {'easy': U_MAZE_EASY_EVAL, 'hard': U_MAZE_HARD_EVAL}
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
        maze_eval_layouts = {'easy': BIG_MAZE_EASY_EVAL, 'hard': BIG_MAZE_HARD_EVAL}
    elif maze_layout_name == "big_maze_sg":
        maze_layout = BIG_MAZE_HARD_EVAL
        maze_eval_layouts = {'easy': BIG_MAZE_EASY_EVAL, 'hard': BIG_MAZE_HARD_EVAL}
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
        maze_eval_layouts = {'easy': HARDEST_MAZE_EASY_EVAL, 'hard': HARDEST_MAZE_HARD_EVAL}
    elif maze_layout_name == "hardest_maze_sg":
        maze_layout = HARDEST_MAZE_HARD_EVAL
        maze_eval_layouts = {'easy': HARDEST_MAZE_EASY_EVAL, 'hard': HARDEST_MAZE_HARD_EVAL}
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    
    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_maze.xml")

    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    eval_possible_goals = {}
    for key, value in maze_eval_layouts.items():
        eval_possible_goals[key] = find_goals(value, maze_size_scaling)

    eval_possible_goals = EvalGoals(eval_goals=eval_possible_goals)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                    j * maze_size_scaling,
                                    MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling,
                                        0.5 * maze_size_scaling,
                                        MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    length = (len(maze_layout) - 1) * maze_size_scaling
    width = (len(maze_layout[0]) - 1) * maze_size_scaling

    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data")
    torso.set("data", f"{robot_x} {robot_y} " + data) 

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    
    return xml_string, possible_goals, eval_possible_goals, (int(length), int(width))

def random_target_fn(possible_goals, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Returns a random target location chosen from possibilities specified in the maze layout."""
    idx = jax.random.randint(rng, (1,), 0, len(possible_goals))
    return rng, jnp.array(possible_goals[idx])[0]

class AntMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        **kwargs,
    ):
        xml_string, possible_goals, eval_possible_goals, (self.x_len, self.y_len) = make_maze(maze_layout_name, maze_size_scaling)
        sys = mjcf.loads(xml_string)
        self.possible_goals = possible_goals
        self.eval_possible_goals = eval_possible_goals

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jnp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset_with_specific_target_settings(self, rng, possible_goals):
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        _, target = random_target_fn(possible_goals, rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        _, target = random_target_fn(self.possible_goals, rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    # Todo rename seed to traj_id
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        success = jnp.array(dist < 0.5, dtype=float)
        success_easy = jnp.array(dist < 2., dtype=float)
        reward = -dist + healthy_reward - ctrl_cost - contact_cost

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel] + [target_pos])