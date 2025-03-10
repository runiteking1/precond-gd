import jax
from jax import tree_map
import jax.numpy as jnp


def tree_dot(a, b):
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        tree_map(lambda x, y: jnp.vdot(x, y), a, b)
    )


def conjugate_gradient(matvec, b, state, x, tol=1e-5, max_iter=1000):
    def body_fn(carry):
        k, r, p, rsold, i = carry
        Ap = matvec(p, state, x)
        alpha = rsold / tree_dot(p, Ap)
        k = tree_map(lambda k, p: k + alpha * p, k, p)
        r = tree_map(lambda r, Ap: r - alpha * Ap, r, Ap)
        rsnew = tree_dot(r, r)
        p = tree_map(lambda r, p: r + (rsnew / rsold) * p, r, p)
        return k, r, p, rsnew, i + 1

    def cond_fn(carry):
        _, _, _, rsnew, i = carry
        return jnp.logical_and(jnp.sqrt(rsnew) >= tol, i < max_iter)

    k = tree_map(jnp.zeros_like, b)  # Initial guess k = 0
    r = tree_map(lambda x: x, b)  # Initial residual r = b - A * k (A * k = 0 initially)
    p = tree_map(lambda x: x, r)  # Initial direction p = r
    rsold = tree_dot(r, r)

    init_carry = (k, r, p, rsold, 0)
    k, r, p, rsnew, i = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    return k, (jnp.sqrt(rsnew), i)
