import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

def plot_results(state, history, x, y):
    # Plot the training loss
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 2, 1)
    plt.semilogy(history['train_loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot the eigenvalues
    plt.subplot(2, 2, 2)
    eigenvalues = np.array(history['eigs'])
    try:
        for i in range(eigenvalues.shape[1]):
            plt.semilogy(eigenvalues[:, i], label=f'Eigenvalue {i + 1}')
    except:
        pass
    plt.xlabel('Iteration (x100)')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of J^T J')
    # plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x, state.apply_fn(state.params, x), 'x-')
    plt.plot(x, y, '.')
    plt.title('Train data versus model output')

    plt.subplot(2, 2, 4)
    # Extract LM results
    indices = [i for i, item in enumerate(history['lm_data']) if item is not None]
    residuals = [item[0] for item in history['lm_data'] if item is not None]
    iterations = [item[1] for item in history['lm_data'] if item is not None]

    # Create a primary axis for residuals
    ax1 = plt.gca()
    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Residual', color=color)
    ax1.semilogy(indices, residuals, color=color, label='Residual')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a secondary y-axis for iteration counts
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Iteration Count', color=color)
    ax2.plot(indices, iterations, color=color, label='Iteration Count')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.title('LM iteration data')
    # plt.show()

    # Desired number of elements
    # step_size = max(1, len(eigevec) // 40)
    # joyplot(
    #     eigevec[::step_size],
    #     figsize=(14,5),
    #     overlap=0.8,
    #     colormap=plt.cm.viridis,
    #     fade=True,
    #     title="Ridge Plot of Largest Eigenvector Over Time",
    #     # labels=[str(i) for i in np.arange(len(output[3]))],
    #     range_style='own'
    # )
    # plt.show()

    # eigenvalues = np.array(eigs_reg)
    # for i in range(eigenvalues.shape[1]):
    #     plt.semilogy(eigenvalues[:, i], label=f'Eigenvalue {i+1}')
    # plt.xlabel('Iteration (x100)')
    # plt.ylabel('Eigenvalue')
    # plt.title('Eigenvalues of J^T J + I')

def plot_results_pinns(state, history, model, x):
    # Plot the training loss
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(history['train_loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Plot the eigenvalues
    plt.subplot(1, 2, 2)
    xs = jnp.concat(
        (x[0], x[1]), axis=0
    )
    u_values = jnp.squeeze(
        model.apply(state.params, xs)
    )
    plt.scatter(
         xs[:, 0], xs[:, 1], c=u_values, cmap='viridis'
    )
    plt.colorbar()

    # Desired number of elements
    # step_size = max(1, len(eigevec) // 40)
    # joyplot(
    #     eigevec[::step_size],
    #     figsize=(14,5),
    #     overlap=0.8,
    #     colormap=plt.cm.viridis,
    #     fade=True,
    #     title="Ridge Plot of Largest Eigenvector Over Time",
    #     # labels=[str(i) for i in np.arange(len(output[3]))],
    #     range_style='own'
    # )
    # plt.show()

    # eigenvalues = np.array(eigs_reg)
    # for i in range(eigenvalues.shape[1]):
    #     plt.semilogy(eigenvalues[:, i], label=f'Eigenvalue {i+1}')
    # plt.xlabel('Iteration (x100)')
    # plt.ylabel('Eigenvalue')
    # plt.title('Eigenvalues of J^T J + I')
