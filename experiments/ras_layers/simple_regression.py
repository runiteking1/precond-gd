import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from matplotlib import pyplot as plt


class MLPLayer(nn.Module):
    """
    One layer of the MLP; multi-dimesionality is untested
    """
    num_units: int

    def setup(self):
        self.dense1 = nn.Dense(self.num_units)
        self.dense2 = nn.Dense(self.num_units)

    def __call__(self, x):
        f = self.dense1(x)
        f = nn.relu(f)
        f = self.dense2(f)
        return f


class SimpleMLP(nn.Module):
    num_layers: int
    num_units: int
    num_classes: int

    def setup(self):
        if self.num_layers >= 100:
            raise Exception(f"num_layers too high; please fix code")

        # Create a list of Dense layers
        self.layers = [
            MLPLayer(self.num_units, name=f"layer_{i:02}") for i in range(self.num_layers)
        ]

    def __call__(self, x):
        # Store input to match the notes
        self.sow('intermediates', f'layer_{0}_output', x)

        # Pass the input through each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Store input to match the notes
            self.sow('intermediates', f'layer_{i + 1}_output', x)

        return x

def f(x: jnp.ndarray):
    """
    Function to fit
    """
    return jnp.sin(4 * x)

# Define the loss function (Mean Squared Error)
def mse_loss(params, x, y):
    predictions = model.apply(params, x).flatten() # Flatten predictions to match y
    return jnp.mean((predictions - y)**2)


if __name__ == "__main__":
    n = 1  # 1 dimensional input and output
    L = 20
    n_samples = 100
    x = jnp.linspace(0, 1, num=100)
    y = f(x)

    # Initialize the model and fit with several method
    # First regular SGD/Adam
    # Second, LM with accurate Jacobian
    # Third, LM with approximate Jacobian

    # SGD/Adam
    model = SimpleMLP(num_layers=L, num_units=n, num_classes=n)
    params = model.init(jax.random.PRNGKey(0), x)

    # Define the optimizer
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop
    num_epochs = 5000
    loss_history = []


    @jax.jit
    def train_step(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(mse_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value


    print("Starting SGD/Adam training...")
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, opt_state, x, y)
        loss_history.append(loss)
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    print("Training finished.")

    # Plot the loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Curve (SGD/Adam)")
    plt.grid(True)

    # Plot the final fitted function against the true function
    plt.subplot(1, 2, 2)
    plt.plot(x, y, label="True Function", color='blue')

    # Get predictions from the trained model
    final_predictions = model.apply(params, x).flatten()
    plt.plot(x, final_predictions, label="Fitted Function (SGD/Adam)", color='red', linestyle='--')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs. Fitted Function")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()