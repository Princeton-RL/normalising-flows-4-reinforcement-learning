import copy
from functools import partial
from typing import Any

import flax
import jax
import distrax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value
from utils.fcnf_networks import RealNVP, RealNVPEncoder

class VINFAgent(flax.struct.PyTreeNode):
    rng: Any
    prior: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        """Compute the VIG critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = jax.lax.stop_gradient( self.sample_actions(batch['next_observations'], seed=sample_rng) )
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        b_noise = self.prior.sample(sample_shape=(batch['observations'].shape[0],), seed=rng)
        b_obs_rep = self.network.select('actor_encoder')( batch['observations'], params=grad_params)
        b_actions, b_logdets = self.network.select('actor')(x=b_noise, y=b_obs_rep, params=grad_params)
        b_actions = jnp.clip(b_actions, -1, 1)

        # Q loss.
        qs = self.network.select('critic')(batch['observations'], actions=b_actions)
        q = jnp.min(qs, axis=0)
        actor_loss = -q.mean()
        if self.config['normalize_q_loss']:
            # # Normalize Q values by the absolute mean to make the loss scale invariant.
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            actor_loss = lam * actor_loss
        else:
            lam=0.0

        # Actor entropy loss
        entropy_loss = (self.prior.log_prob(b_noise) - b_logdets).mean()

        # Actor BC loss
        p_noise, p_logdets = self.network.select('actor')(x=batch['actions'], y=b_obs_rep, reverse=True, params=grad_params)
        bc_loss = - (self.prior.log_prob(p_noise) + p_logdets).mean() * self.config['alpha_actor'] 
        
        total_loss = actor_loss + bc_loss + entropy_loss * self.config['alpha_actor_entropy']

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'mse': jnp.square(b_actions - batch['actions']).sum(axis=-1).mean(),
            'lam': lam,
            'p_norm' : jnp.square(p_noise).mean(),
            'actor_entropy': -entropy_loss,
        }
        
    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = critic_loss + actor_loss
        return loss, info
        
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('full_update',))
    def update(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            # Update the target networks only when `full_update` is True.
            self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        b_noise = self.prior.sample(sample_shape=(observations.shape[0],), seed=seed)
        obs_rep = self.network.select('actor_encoder')(observations)
        actions, _ = self.network.select('actor')(x=b_noise, y=obs_rep)
        actions = jnp.clip(actions, -1, 1)
        return actions
        
    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )

        actor_encoder_def = RealNVPEncoder(
            input_size = ex_observations.shape[-1],
            rep_size = config['bc_rep_size']
        )
        actor_def = RealNVP(
            num_blocks = config['bc_num_blocks'], 
            in_channels = action_dim, 
            cond_channels = config['bc_rep_size'],
            channels = config['bc_channels'],
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_encoder=(actor_encoder_def, (ex_observations,)),
            actor=(actor_def, ( ex_actions, jnp.zeros(shape=(1, config['bc_rep_size'])) )),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        loc = jnp.zeros(ex_actions.shape[-1], dtype=jnp.float32)
        cov = jnp.eye(ex_actions.shape[-1], dtype=jnp.float32)
        prior = distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)

        return cls(rng, prior=prior, network=network, config=flax.core.FrozenDict(**config))

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='vinf',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.

            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values.
            alpha_actor=0.0,  # Actor BC coefficient.
            alpha_actor_entropy=0.0,

            bc_rep_size = 64,
            bc_num_blocks = 6,
            bc_channels = 256,

            actor_noise=0.2,  # Actor noise scale.
            
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config