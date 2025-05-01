import jax
import jax.numpy as jnp


def batch_data(x, y, batch_size, rng_key):
    """Yield batches of data."""
    num_samples = x.shape[0]
    indices = jax.random.permutation(rng_key, num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield x[batch_indices], y[batch_indices]


def batch_data_pinns(x, batch_size, rng_key):
    """
    Yield batches of data for PINNs.
    If batch_size < 0, do full batch without permutation.
    """
    x_int, x_bnd = x
    num_samples_int = x_int.shape[0]
    num_samples_bnd = x_bnd.shape[0]
    ratio = num_samples_int / num_samples_bnd

    if batch_size < 0:
        # Full batch mode, no permutation needed
        yield x_int, x_bnd
        return

    # Permute indices for minibatch mode
    indices_int = jax.random.permutation(rng_key, num_samples_int)
    rng_key, _ = jax.random.split(rng_key)
    indices_bnd = jax.random.permutation(rng_key, num_samples_bnd)

    batch_size_int = int(jnp.round(batch_size * ratio / (1 + ratio)))
    batch_size_bnd = batch_size - batch_size_int

    batch_size_int = max(batch_size_int, 1)
    batch_size_bnd = max(batch_size_bnd, 1)

    num_batches = max(num_samples_int // batch_size_int, num_samples_bnd // batch_size_bnd)
    for i in range(num_batches):
        start_idx_int = i * batch_size_int
        end_idx_int = min(start_idx_int + batch_size_int, num_samples_int)
        batch_indices_int = indices_int[start_idx_int:end_idx_int]

        start_idx_bnd = i * batch_size_bnd
        end_idx_bnd = min(start_idx_bnd + batch_size_bnd, num_samples_bnd)
        batch_indices_bnd = indices_bnd[start_idx_bnd:end_idx_bnd]

        yield x_int[batch_indices_int], x_bnd[batch_indices_bnd]


def generate_training_data(num_interior: int, num_boundary: int, seed: int=0):
    """
    Generate training data

    Args:
        num_interior (int): number of interior points
        num_boundary (int): number of boundary points
        seed (int, optional): Seed. Defaults to 0.
    """
    key = jax.random.PRNGKey(seed)

    # Generate interior points
    key, subkey = jax.random.split(key)
    interior_points = jax.random.uniform(subkey, (num_interior, 2))

    # Generate boundary points
    key, subkey = jax.random.split(key)
    boundary_points_1 = jax.random.uniform(subkey, (num_boundary // 4, 2)) * jnp.array([1, 0])
    key, subkey = jax.random.split(key)
    boundary_points_2 = jax.random.uniform(subkey, (num_boundary // 4, 2)) * jnp.array([1, 0]) + jnp.array([0, 1])
    key, subkey = jax.random.split(key)
    boundary_points_3 = jax.random.uniform(subkey, (num_boundary // 4, 2)) * jnp.array([0, 1])
    key, subkey = jax.random.split(key)
    boundary_points_4 = jax.random.uniform(subkey, (num_boundary // 4, 2)) * jnp.array([0, 1]) + jnp.array([1, 0])

    boundary_points = jnp.concatenate([boundary_points_1, boundary_points_2, boundary_points_3, boundary_points_4], axis=0)

    return interior_points, boundary_points
