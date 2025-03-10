import jax
import jax.numpy as jnp


def compute_jacobian(params, model, x):
    jacobian_fn = jax.jacrev(model.apply, argnums=0)
    return jacobian_fn(params, x)


# Might be able to use ravel eventually but this is easier
def flatten_jacobian(jacobian_dict, x):
    batch_size = x.shape[0]
    flat_jacobian = []
    for layer in jacobian_dict['params'].values():
        for param in layer.values():
            flat_jacobian.append(param.reshape(batch_size, -1))
    return jnp.concatenate(flat_jacobian, axis=1)


# Might be able to use ravel eventually but this is easier
def flatten_grad(grad_dict, x):
    """

    :param grad_dict:
    :param x: Sample batch data
    :return:
    """
    batch_size = x.shape[0]
    flat_grad = []
    for layer in grad_dict['params'].values():
        for param in layer.values():
            flat_grad.append(param.reshape(-1))
    return jnp.concatenate(flat_grad)
