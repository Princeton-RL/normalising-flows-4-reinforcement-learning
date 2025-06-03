import jax
import time
import numpy as np

from flax import struct
from typing import Dict
from envs.wrappers import EvalWrapper, EvalMetrics

@struct.dataclass
class HierarchicalEvalMetrics:
	metrics: Dict[str, EvalMetrics]

def generate_unroll(actor_step, training_state, env, env_state, unroll_length, extra_fields=()):
	"""Collect trajectories of given unroll_length."""
	@jax.jit
	def f(carry, unused_t):
		state = carry
		nstate, transition = actor_step(training_state, env, state, extra_fields=extra_fields)
		return nstate, transition

	final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
	return final_state, data

class CrlEvaluator():

	def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
		self._key = key
		self._eval_walltime = 0.
		eval_env = EvalWrapper(eval_env)
		eval_env.reset = jax.jit(eval_env.reset)
		eval_env.step = jax.jit(eval_env.step)

		if hasattr(eval_env, 'eval_possible_goals'):
			eval_env.reset_with_specific_target_settings = jax.jit(eval_env.reset_with_specific_target_settings)
			self.eval_possible_goals = eval_env.eval_possible_goals

			def generate_eval_unroll(training_state, key):
				metrics = {}
				for dict_key, dict_value in self.eval_possible_goals.eval_goals.items():
					eval_key, key = jax.random.split(key, 2)
					reset_keys = jax.random.split(eval_key, num_eval_envs)
					eval_first_state = eval_env.reset_with_specific_target_settings(reset_keys, dict_value)
					metrics[dict_key] = generate_unroll(
						actor_step,
						training_state,
						eval_env,
						eval_first_state,
						unroll_length=episode_length)[0].info["eval_metrics"]

				return HierarchicalEvalMetrics(metrics=metrics)

			self._generate_eval_unroll = jax.jit(generate_eval_unroll)
			self._steps_per_unroll = episode_length * num_eval_envs * len(self.eval_possible_goals.eval_goals.keys())

		else:
			def generate_eval_unroll(training_state, key):
				reset_keys = jax.random.split(key, num_eval_envs)
				eval_first_state = eval_env.reset(reset_keys)
				return generate_unroll(
					actor_step,
					training_state,
					eval_env,
					eval_first_state,
					unroll_length=episode_length)[0]

			self._generate_eval_unroll = jax.jit(generate_eval_unroll)
			self._steps_per_unroll = episode_length * num_eval_envs

	def run_evaluation(self, training_state, training_metrics, aggregate_episodes = True):
		"""Run one epoch of evaluation."""
		self._key, unroll_key = jax.random.split(self._key)
		t = time.time()

		if hasattr(self, 'eval_possible_goals'):
			h_metrics = self._generate_eval_unroll(training_state, unroll_key)
			h_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), h_metrics)
			epoch_eval_time = time.time() - t
			
			metrics = {}
			for key_eval_metrics, eval_metrics in h_metrics.metrics.items():
				
				aggregating_fns = [(np.mean, ""),]
				for (fn, suffix) in aggregating_fns:
					metrics.update(
						{
							f"eval/{key_eval_metrics}/episode_{name}{suffix}": (
								fn(eval_metrics.episode_metrics[name]) if aggregate_episodes else eval_metrics.episode_metrics[name]
							)
							for name in ['reward', 'success', 'success_easy', 'success_hard', 'success_stand', 'dist', 'distance_from_origin'] if name in eval_metrics.episode_metrics.keys()
						}
					)

				# We check in how many env there was at least one step where there was success
				if "success" in eval_metrics.episode_metrics:
					metrics[f"eval/{key_eval_metrics}/episode_success_any"] = np.mean(
						eval_metrics.episode_metrics["success"] > 0.0
					)

			metrics[f"eval/{key_eval_metrics}/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
			metrics["eval/epoch_eval_time"] = epoch_eval_time
			metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
			self._eval_walltime = self._eval_walltime + epoch_eval_time
			metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

		else:
			eval_state = self._generate_eval_unroll(training_state, unroll_key)
			eval_metrics = eval_state.info["eval_metrics"]
			eval_metrics.active_episodes.block_until_ready()
			epoch_eval_time = time.time() - t
			metrics = {}
			aggregating_fns = [(np.mean, ""),]

			for (fn, suffix) in aggregating_fns:
				metrics.update(
					{
					f"eval/episode_{name}{suffix}": (
						fn(eval_metrics.episode_metrics[name]) if aggregate_episodes else eval_metrics.episode_metrics[name]
					)
					for name in ['reward', 'success', 'success_easy', 'success_hard', 'success_stand', 'dist', 'distance_from_origin'] if name in eval_metrics.episode_metrics.keys()
					}
				)

			# We check in how many env there was at least one step where there was success
			if "success" in eval_metrics.episode_metrics:
				metrics["eval/episode_success_any"] = np.mean(
					eval_metrics.episode_metrics["success"] > 0.0
				)

			metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
			metrics["eval/epoch_eval_time"] = epoch_eval_time
			metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
			self._eval_walltime = self._eval_walltime + epoch_eval_time
			metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}
		return metrics