import jax
import jax.numpy as jnp

from flax import struct
from brax.envs.base import Wrapper
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
    
# from https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py

class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""
    def __init__(self, env, episode_length, action_repeat):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state
    
    def reset_with_specific_target_settings(self, rng, possible_goals):
        state = self.env.reset_with_specific_target_settings(rng, possible_goals)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state
    
    def step(self, state, action):
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps
        return state.replace(done=done)

class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""
    def __init__(self, env, batch_size = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng):
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def reset_with_specific_target_settings(self, rng, possible_goals):
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset_with_specific_target_settings, in_axes=(0, None))(rng, possible_goals)

    def step(self, state, action: jax.Array):
        return jax.vmap(self.env.step)(state, action)
    
class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state
    
    def reset_with_specific_target_settings(self, rng, possible_goals):
        state = self.env.reset_with_specific_target_settings(rng, possible_goals)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state, action):
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )

        obs = where_done(state.info['first_obs'], state.obs)
        
        return state.replace(pipeline_state=pipeline_state, obs=obs)
  
@struct.dataclass
class EvalMetrics:
  """Dataclass holding evaluation metrics for Brax.

  Attributes:
      episode_metrics: Aggregated episode metrics since the beginning of the episode.
      active_episodes: Boolean vector tracking which episodes are not done yet.
      episode_steps: Integer vector tracking the number of steps in the episode.
  """

  episode_metrics: Dict[str, jax.Array]
  active_episodes: jax.Array
  episode_steps: jax.Array

class EvalWrapper(Wrapper):
    """Brax env with eval metrics."""	
    def reset(self, rng):
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def reset_with_specific_target_settings(self, rng, possible_goals):
        reset_state = self.env.reset_with_specific_target_settings(rng, possible_goals)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, state, action):
        state_metrics = state.info['eval_metrics']
        
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f'Incorrect type for state_metrics: {type(state_metrics)}')
        del state.info['eval_metrics']

        nstate = self.env.step(state, action)
        nstate.metrics['reward'] = nstate.reward
        
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info['steps'],
            state_metrics.episode_steps,
        )
        
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )

        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )

        nstate.info['eval_metrics'] = eval_metrics
        return nstate