import jax


def batch_data(x, y, batch_size, rng_key):
    """Yield batches of data."""
    num_samples = x.shape[0]
    indices = jax.random.permutation(rng_key, num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield x[batch_indices], y[batch_indices]
