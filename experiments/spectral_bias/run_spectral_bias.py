from functools import partial
from typing import Union

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState  # Useful dataclass to keep train state
from jax.tree import map as tree_map
from tqdm import tqdm

from cg import conjugate_gradient
from data_utils import batch_data
from jacobian_tools import flatten_jacobian
from models.MLP import *
from plotting import plot_results

def power_iteration(A, num_simulations=10):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    A = A.data
    b_k = A.new(A.shape[1], 1).normal_()
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k
        # calculate the norm
        b_k1_norm = torch.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    return ((b_k.t() @ A @ b_k) / b_k.t() @ b_k).squeeze().abs()
    # return torch.dot(torch.dot(b_k.t(), A), b_k) / torch.dot(b_k.t(), b_k)

def spectral_norm(model):
    norms = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            if layer.in_features == layer.out_features:
                norms.append(power_iteration(layer.weight).cpu().numpy())
            elif layer.in_features == 1 or layer.out_features == 1:
                norms.append(torch.norm(layer.weight.data))
    return norms

def train_model(state: TrainState, x, y, num_iterations: int = 1_000,):
    @jax.jit
    def model_output(params, x):
        # f(\theta, x); jax defaults to derivs of first component
        return state.apply_fn(params, x)

    @jax.jit
    def compute_manual_gradient(state, x, y):
        # For MSE
        n = x.shape[0]

        # Compute the value of f(\theta, x) and the function for VJP
        preds, vjp_fn = jax.vjp(model_output, state.params, x)
        residuals = (preds - y)

        # Explicitly calculate loss for metrics
        loss = 1 / n * jnp.sum(residuals ** 2)

        # residuals^T jacobian (Might need transpose here in 2D)
        manual_grad = vjp_fn(residuals * 2 / n)[0]  # First term is wrt params

        return manual_grad, loss, vjp_fn

    @jax.jit
    def train_step(state, x, y):
        # Manually calculate the gradient to demonstrate
        grads, loss, vjp_fn = compute_manual_gradient(state, x, y)
        # Apply the gradient update
        state = state.apply_gradients(grads=grads)

        return state, loss

    # Recording
    metrics_history = {
        'train_loss': [], 'eigs': [], 'mat': [],
        'frames': [],
    }



    for epoch in tqdm(range(num_iterations), leave=True):
        state, loss, lm_data = train_step(state, x, y)

        # Compute the average loss for the epoch
        metrics_history['train_loss'].append(loss)

        if epoch % 100 == 0:
            # Measure spectral norm
            frames.append(
                Namespace(iter_num=iter_num,
                                    prediction=y.data.cpu().numpy(),
                                    loss=loss.item(),
                                    spectral_norms=spectral_norm(model))
            )
    # Done
    model.eval()
    return frames