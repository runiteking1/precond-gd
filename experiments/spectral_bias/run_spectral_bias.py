from functools import partial, reduce
from typing import List, Union

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
from model_utils import create_train_state
from MLP import MLP_1D
from jax import random

from dataclasses import field
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import numpy as np


@chex.dataclass
class Options:
    # Data Generation
    N: int = 200
    K: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    A: List[int] = field(default_factory=lambda: [1 for _ in range(10)])
    INP_DIM: int = 1
    OUT_DIM: int = 1
    WIDTH: int = 256
    DEPTH: int = 6
    NUM_ITER: int = 60000
    REC_FRQ: int = 100
    LR: float = 0.0003
    def __post_init__(self):
        object.__setattr__(self, 'PHI', [np.random.rand() for _ in self.K])


@chex.dataclass(frozen=True)
class Frame:
    iter_num: int
    prediction: jnp.ndarray
    loss: float
    spectral_norms: list


def power_iteration(A, num_simulations=10, key=random.PRNGKey(0)):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = random.normal(key, (A.shape[1], 1))
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = jnp.dot(A, b_k)
        # calculate the norm
        b_k1_norm = jnp.linalg.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    return jnp.abs((jnp.dot(b_k.T, jnp.dot(A, b_k)) / jnp.dot(b_k.T, b_k)).squeeze())

# def spectral_norm(model):
#     norms = []
#     for layer in model:
#         if isinstance(layer, nn.Linear):
#             if layer.in_features == layer.out_features:
#                 norms.append(power_iteration(layer.weight).cpu().numpy())
#             elif layer.in_features == 1 or layer.out_features == 1:
#                 norms.append(torch.norm(layer.weight.data))
#     return norms

def spectral_norm(params, key=random.PRNGKey(0)):
    """Calculates spectral norms for specific layers in a Flax model's parameters."""
    norms = []
    params = params['params'] # Extract out the params dictionary
    for layer_name, layer_params in params.items():
        if 'kernel' in layer_params: # Linen linear layers use 'kernel' instead of 'weight'
            weight = layer_params['kernel']
            if weight.shape[0] == weight.shape[1]:
                norms.append(power_iteration(weight))
            elif weight.shape[0] == 1 or weight.shape[1] == 1:
                norms.append(jnp.linalg.norm(weight))
        # print(f'{layer_params=}')
    # print(norms)
    return norms

def fft(opt, yt):
    n = len(yt)  # length of the signal
    k = jnp.arange(n)
    T = n / opt.N
    frq = k / T  # two sides frequency range
    frq = frq[:n // 2]  # one side frequency range
    # -------------
    FFTYT = jnp.fft.fft(yt) / n  # fft computing and normalization
    FFTYT = FFTYT[:n // 2]
    fftyt = jnp.abs(FFTYT)
    return frq, fftyt

def compute_spectra(opt, frames): 
    # Make array for heatmap
    dynamics = []
    xticks = []
    for iframe, frame in enumerate(frames): 
        # Compute fft of prediction
        frq, yfft = fft(opt, frame.prediction.squeeze())
        dynamics.append(yfft)
        xticks.append(frame.iter_num)
    return jnp.array(frq), jnp.array(dynamics), jnp.array(xticks)

def plot_spectral_dynamics(opt, all_frames):
    all_dynamics = []
    # Compute spectra for all frames
    for frames in all_frames: 
        frq, dynamics, xticks = compute_spectra(opt, frames)
        all_dynamics.append(dynamics)
    print(frq, dynamics)
    # Average dynamics over multiple frames
    # mean_dynamics.shape = (num_iterations, num_frequencies)
    mean_dynamics = jnp.array(all_dynamics).mean(0)
    # Select the frequencies which are present in the target spectrum
    freq_selected = mean_dynamics[:, jnp.sum(frq.reshape(-1, 1) == jnp.array(opt.K).reshape(1, -1), 
                                            axis=-1, dtype='bool')]
    # Normalize by the amplitude. Remember to account for the fact that the measured spectra 
    # are single-sided (positive freqs), so multiply by 2 accordingly
    norm_dynamics = 2 * freq_selected / jnp.array(opt.A).reshape(1, -1)
    # Plot heatmap
    plt.figure(figsize=(7, 6))
    # plt.title("Evolution of Frequency Spectrum (Increasing Amplitudes)")
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels=opt.K, 
                yticklabels=[(frame.iter_num if frame.iter_num % 10000 == 0 else '') 
                             for _, frame in zip(range(norm_dynamics.shape[0]), frames)][::-1], 
                vmin=0., vmax=1., 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Training Iteration")
    plt.show()

def train_model(state: TrainState, x, y, num_iterations: int = 10_000, opt=None,
                precond=False):
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

        return manual_grad, loss, vjp_fn, preds

    @jax.jit
    def train_step(state, x, y):
        # Manually calculate the gradient to demonstrate
        grads, loss, _, preds = compute_manual_gradient(state, x, y)

        if precond: 
            jacobian_fn = flatten_jacobian(jax.jacrev(model_output)(state.params, x), x)

            # Compute SVD
            _, s, vh = jnp.linalg.svd(jacobian_fn, full_matrices=False)

            # Compute inverse
            threshold = 1e-3
            s_inv = jnp.where(s > 1e-3, s ** -2, threshold)
            # print(s_inv.shape)

            # Apply to grads
            # print(flatten_grad(grads))
            flattened, back = jax.flatten_util.ravel_pytree(grads)
            # print(flattened)
            my_change = vh.T @ jnp.diag(s_inv) @ vh @ flattened
            grads = back(my_change)


        # Apply the gradient update
        state = state.apply_gradients(grads=grads)

        return state, loss, preds

    # Recording
    metrics_history = {
        'train_loss': [], 'eigs': [], 'mat': [],
        'frames': [],
    }


    for epoch in tqdm(range(num_iterations), leave=True):
        state, loss, preds = train_step(state, x, y)

        # Compute the average loss for the epoch
        metrics_history['train_loss'].append(loss)

        if epoch % 100 == 0:
            # Measure spectral norm
            metrics_history['frames'].append(
                Frame(iter_num=epoch, 
                        prediction=preds, 
                        loss=loss, 
                        spectral_norms=spectral_norm(state.params))
            )

    return metrics_history

def make_phased_waves(opt):
    t = jnp.arange(0, 1, 1./opt.N)
    if opt.A is None:
        yt = reduce(lambda a, b: a + b, 
                    [jnp.sin(2 * jnp.pi * ki * t + 2 * jnp.pi * phi) for ki, phi in zip(opt.K, opt.PHI)])
    else:
        yt = reduce(lambda a, b: a + b, 
                    [Ai * jnp.sin(2 * jnp.pi * ki * t + 2 * jnp.pi * phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])
    return t, yt

def plot_wave_and_spectrum(opt, x, yox):
    # Btw, "yox" --> "y of x"
    # Compute fft
    k, yok = fft(opt, yox)
    # Plot
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))
    ax0.set_title("Function")
    ax0.plot(x, yox)
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")
    ax1.set_title("FT of Function")
    ax1.plot(k, yok)
    ax1.set_xlabel("k")
    ax1.set_ylabel("f(k)")
    plt.show()

def run_exp():
    opt = Options()
    opt.N = 200
    opt.LR = 1e-3
    x, y = make_phased_waves(opt)
    # plot_wave_and_spectrum(opt, x, y)

    # print(model.tabulate(
    #     jax.random.key(0), x, compute_flops=True,
    #     compute_vjp_flops=True)
    # )
    all_frames = []
    repeats = 1
    for seed in range(repeats): 
        # Replace with new phi
        opt.PHI = [np.random.rand() for _ in opt.K]
        x, y = make_phased_waves(opt)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # x = jnp.linspace(0, 1, 200).reshape(-1, 1)
        # y = jnp.sin(1 * jnp.pi * x) + jnp.sin(5 * jnp.pi * x)

        model = MLP_1D(num_layers=1, hidden_dim=16)
        key = jax.random.PRNGKey(seed)

        state = create_train_state(model, key, learning_rate=opt.LR, momentum=0, optimizer='sgd')
        metrics_history = train_model(state, x, y, num_iterations=60000, opt=opt, 
                                      precond=True
                                      
                                             )
        all_frames.append(metrics_history['frames'])
        plt.plot(metrics_history['train_loss'])
        plt.show()
        # print(metrics_history['frames'][0:5])
    # # print(metrics_history['frames'])
    # # plot_spectral_dynamics(opt, [metrics_history['frames']])
    plot_spectral_dynamics(opt, all_frames)

if __name__ == '__main__':
    print('Running experiment 1.')
    run_exp()
