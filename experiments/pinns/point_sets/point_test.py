import os
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from run_pinns import *


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--training', choices=['sgd', 'smw', 'gn'], required=True)
args = parser.parse_args()
training_mode = args.training



os.environ['JAX_PLATFORM_NAME'] = 'gpu'
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')


# Settings
base_path = f"./{training_mode}"
os.makedirs(base_path, exist_ok=True)

# Sweep parameters
interior_scales = [1, 2, 4, 12]  # scale num_interior
boundary_scales = [1, 2, 4, 12]     # scale num_boundary
NM = [#(1,1),
      (2,2)]          # sin(pi x) sin(pi y) frequencies
dims = 256                  # model width
batch_size = -1             # full batch (-1 means full batch)

# Global training settings
if training_mode == 'sgd':
    lr = 1e-3
    lm_info = None
    num_iterations = 50_000
elif training_mode == 'smw':
    lr = 2e-2
    lm_info = PrecondData(method='smw', epoch=0, lamb=1e-1)
    num_iterations = 10_000
elif training_mode == 'gn':
    lr = 5e-3
    lm_info = PrecondData(method='gn-exact', epoch=0, thresh=1e-3)
    num_iterations = 5_000
else:
    raise ValueError(f"Unknown training mode {training_mode}")


# Functions
def get_forcing_fxn(n, m):
    return lambda x: (n**2 + m**2) * (jnp.pi**2) * jnp.sin(n * jnp.pi * x[0]) * jnp.sin(m * jnp.pi * x[1])

def get_solution_fxn(n, m):
    return lambda x: jnp.sin(n * jnp.pi * x[0]) * jnp.sin(m * jnp.pi * x[1])

# Storage
metrics_history_sgd = {}

# Main loop
for int_scale in interior_scales:
    for bnd_scale in boundary_scales:
        for n, m in NM:
            # Problem setup
            num_interior = 20 * int_scale
            num_boundary = 20 * bnd_scale
            interior_points, boundary_points = generate_training_data(num_interior, num_boundary)
            problem_data = (interior_points, boundary_points)

            # Unique run name
            name = f"{base_path}/fx_{n}_fy_{m}_int{num_interior}_bnd{num_boundary}"
            metrics_file = name + ".pkl"

            # Check if already computed
            if os.path.exists(metrics_file):
                print(f"Loading existing metrics for {name}")
                metrics_history_sgd[name] = load_metrics(metrics_file)
                continue

            print(f"Training {name}...")

            # Model and optimizer
            model = MLP_1D(1, dims)
            dummy_input = jnp.ones((1, 2))
            state = create_train_state(model, learning_rate=lr, momentum=0, optimizer='sgd', shape=dummy_input)

            # Train
            state, metrics = train_model(
                state,
                problem_data=problem_data,
                forcing_function=get_forcing_fxn(n, m),
                solution_function=get_solution_fxn(n, m),
                num_iterations=num_iterations,
                lm_schedule=lm_info,
                batch_size=batch_size,
                metrics_file=metrics_file,
            )

            metrics_history_sgd[name] = metrics

print("All runs complete.")

