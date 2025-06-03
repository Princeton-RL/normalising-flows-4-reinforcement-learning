import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

import flax.linen as nn
import optax
import distrax
from flax import struct
from flax.training import train_state
from functools import partial

def create_prior(input_dims: int):
    loc = jnp.zeros(input_dims, dtype=jnp.float32)
    cov = jnp.eye(input_dims, dtype=jnp.float32)
    return distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)

def count_parameters(params):
    return sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_leaves(params))

class InvertiblePLU(nn.Module):
    features: int
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
    
    def setup(self):
        d = self.features
        key = self.key
        
        w_shape = (d, d)
        w_init = nn.initializers.orthogonal()(key, w_shape)
        P, L, U = jax.scipy.linalg.lu(w_init)
        s = jnp.diag(U)
        U = U - jnp.diag(s)

        self.P = P
        self.P_inv = jax.scipy.linalg.inv(P)
        
        self.L_init = jnp.tril(L, k=-1)
        self.U_init = jnp.triu(U, k=1)
        self.s_init = s

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        d = self.features

        L_free = self.param("L", lambda rng: self.L_init)
        U_free = self.param("U", lambda rng: self.U_init)
        
        L = jnp.tril(L_free, k=-1) + jnp.eye(d)
        U = jnp.triu(U_free, k=1)
        s = self.param("s", lambda rng: self.s_init)
        
        W = self.P @ L @ (U + jnp.diag(s))

        if not reverse:
            z = jnp.dot(x, W)
            logdet = jnp.sum(jnp.log(jnp.abs(s)))
            return z , jnp.expand_dims( logdet, 0)
        
        else:
            
            U_inv = jax.scipy.linalg.solve_triangular(U + jnp.diag(s), jnp.eye(self.features), lower=False)
            L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(self.features), lower=True, unit_diagonal=True)
            
            W_inv = U_inv @ L_inv @ self.P_inv
            
            z = jnp.dot(x, W_inv)
            logdet = jnp.sum(jnp.log(jnp.abs(s)))
            return z, -jnp.expand_dims( logdet, 0)

class MetaBlock(nn.Module):
    in_channels: int
    channels: int
    cond_channels: int
    # flip: bool
    block_idx: int

    def setup(self): 
        def kernel_init(key, shape, dtype=jnp.float32):
            in_features = shape[0]
            k = jnp.sqrt( 1.0 / in_features )
            return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)

        def bias_init(key, shape, dtype, in_features):
            # print(shape)
            k = jnp.sqrt( 1.0 / in_features )
            return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)
        
        self.l = InvertiblePLU(
            features = self.in_channels
        )
        
        self.t = nn.Sequential([
            nn.Dense(self.channels + self.cond_channels, kernel_init=kernel_init, bias_init=partial(bias_init, in_features=self.cond_channels+self.in_channels)), nn.leaky_relu, nn.LayerNorm(),
            nn.Dense(self.channels, kernel_init=kernel_init, bias_init=partial(bias_init, in_features=self.channels)), nn.leaky_relu, nn.LayerNorm(),
            # nn.Dense((in_channels // 2), kernel_init=kernel_init, bias_init=partial(bias_init, in_features=self.channels)), nn.tanh
            nn.Dense( (self.in_channels // 2), kernel_init=jax.nn.initializers.zeros )
        ])

        self.s = nn.Sequential([
            nn.Dense(self.channels + self.cond_channels, kernel_init=kernel_init, bias_init=partial(bias_init, in_features=self.cond_channels+self.in_channels)), nn.leaky_relu, nn.LayerNorm(),
            nn.Dense(self.channels, kernel_init=kernel_init, bias_init=partial(bias_init, in_features=self.channels)), nn.leaky_relu, nn.LayerNorm(),
            # nn.Dense((in_channels // 2), kernel_init=kernel_init, bias_init=partial(bias_init, in_features=self.channels)), nn.tanh
            nn.Dense( (self.in_channels // 2), kernel_init=jax.nn.initializers.zeros )
        ])

    def __call__(self, x, y, reverse=False):
        if not reverse:
            return self.forward(x, y)
        else:
            return self.reverse(x, y)
        
    def forward(self, x, y):  
        # if self.flip: 
        #     x = jnp.flip(x, axis=1)

        x, log_det = self.l(x)
            
        x_cond, x_trans = jnp.array_split(x, 2, axis=1)
        s = self.s( jnp.concatenate([x_cond, y], axis=-1) ) 
        t = self.t( jnp.concatenate([x_cond, y], axis=-1) )
        x_trans = (x_trans - t) * jnp.exp(-s)
        x = jnp.concatenate((x_cond, x_trans), axis=1)
        log_det += -jnp.sum(s, axis=1)
        
        # if self.flip: 
        #     x = jnp.flip(x, axis=1)
            
        return x, log_det

    def reverse(self, z, y):
        # if self.flip: 
        #     z = jnp.flip(z, axis=1)

        z_cond, z_trans = jnp.array_split(z, 2, axis=1)
        s = self.s( jnp.concatenate([z_cond, y], axis=-1) ) 
        t = self.t( jnp.concatenate([z_cond, y], axis=-1) )
        z_trans = z_trans * jnp.exp(s) + t
        z = jnp.concatenate((z_cond, z_trans), axis=1)

        z, log_det = self.l(z, reverse=True)

        log_det += jnp.sum(s, axis=1)

        # if self.flip: 
        #     z = jnp.flip(z, axis=1)

        return z, log_det
    
class RealNVP(nn.Module):
    num_blocks: int
    in_channels: int
    cond_channels: int
    channels: int
    
    def setup(self):
        self.blocks = [
            MetaBlock(
                in_channels=self.in_channels,
                cond_channels=self.cond_channels,
                channels=self.channels,
                # flip=i%2==0,
                block_idx=i,
            )
            for i in range(self.num_blocks)
        ]

    def __call__(self, x, y, reverse=False):
        if not reverse:
            return self.forward(x, y)
        else:
            return self.reverse(x, y)

    def forward(self, x, y):
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for block in self.blocks:
            x, log_det = block(x, y)
            log_dets = log_dets + log_det
        return x, log_dets
        
    def reverse(self, x, y):
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for block in reversed(self.blocks):
            x, log_det = block(x, y, reverse=True)
            log_dets = log_dets + log_det
        return x, log_dets
    
class RealNVPEncoder(nn.Module):
    input_size: int
    rep_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        # Final projection to rep_size
        x = nn.Dense(self.rep_size)(x)
        return x