import jax
import optax
from jax import numpy as jnp

from flax.training.train_state import TrainState


def create_train_state(module, rng=None, learning_rate=1e-3, momentum=.9, optimizer='adam'):
    # Can be more careful with RNG; but this should be fine
    if rng is None:
        rng = jax.random.key(0)

    # Init parameters
    params = module.init(rng, jnp.ones([1]))

    # Init optimizer
    if optimizer == 'sgd':
        tx = optax.sgd(learning_rate, momentum)
    elif optimizer == 'adamw':
        tx = optax.adamw(learning_rate)
    elif optimizer == 'adam':
        tx = optax.adam(learning_rate)
    else:
        tx = optax.sgd(learning_rate)

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx
    )
