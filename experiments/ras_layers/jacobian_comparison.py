import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')

"""
There's some issues with conditioning; have to use relative
"""


import jax.numpy as jnp
from flax import linen as nn
import numpy as np

class MLP(nn.Module):
    """
    We first deal with something that's not exactly MLP, but close enough
    """
    num_units: int
    
    def setup(self):
        self.dense1 = nn.Dense(self.num_units, dtype=jnp.float64)
        # self.dense2 = nn.Dense(self.num_units)
    
    def __call__(self, x):
        f = self.dense1(x)
        f = nn.leaky_relu(f)
        # f = nn.tanh(f)
        # x = self.dense2(x)
        return x + f
    

class SimpleMLP(nn.Module):
    num_layers: int
    num_units: int
    num_classes: int

    def setup(self):
        # Create a list of Dense layers
        self.layers = [MLP(self.num_units) for _ in range(self.num_layers)]
        # self.classification_layer = nn.Dense(self.num_classes)

    def __call__(self, x):
        # Store input to match the notes
        self.sow('intermediates',  f'layer_{0}', x)
        
        # Pass the input through each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Store input to match the notes
            self.sow('intermediates', f'layer_{i+1}', x)
            
        # Final layer to produce output
        # x = self.classification_layer(x)
        return x

def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

if __name__ == "__main__": 
    n = 1 # 1 dimensional input and output 
    L = 64 # Number of layers; trying to emulate the paper
    n_samples = 1

    model = SimpleMLP(num_layers=L, num_units=n, num_classes=n)
    rng = jax.random.PRNGKey(0)
    input_shape = (n_samples, n)  # Batch size of 1 and input dimension of 32
    dummy_input = jnp.ones(input_shape, dtype=jnp.float64)
    params = model.init(rng, dummy_input)

    x = jnp.ones((n_samples, n), dtype=jnp.float64) 
    y = jnp.ones((n_samples, n), dtype=jnp.float64) * .1
    prediction = model.apply(params, x)

    def loss_fn(params): 
        prediction = model.apply(params, x)
        loss = mse_loss(prediction, y)
        return loss
    print(params['params']['layers_0'], end='\n\n')

    # First way is to just use jax.grad
    grad_fn = jax.grad(loss_fn)
    real_grad = grad_fn(params)

    # Second way is to just use vjp 
    _, vjp_fn = jax.vjp(model.apply, params, x)
    residuals = (prediction - y)
    manual_grad = vjp_fn(residuals * 2 / (n * n_samples))[0] 
    print(manual_grad['params']['layers_0'], end='\n\n')

    # Last way, obtain Jacobian, then multiply myself
    # Compute the Jacobian matrix of the model's output with respect to the input
    jacobian = jax.jacobian(model.apply, argnums=0)(params, x)
    treemap_grad = jax.tree.map(lambda x: jnp.tensordot(residuals * 2 / (n * n_samples), x), jacobian)
    print(treemap_grad['params']['layers_0'])

    l2_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), real_grad)
    true_l2_norm = jax.tree.reduce(lambda x, y: x + y, l2_norms)

    difference_tree = jax.tree.map(lambda x, y: x - y, manual_grad, real_grad)
    l2_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), difference_tree)
    total_l2_norm = jax.tree.reduce(lambda x, y: x + y, l2_norms)
    print(total_l2_norm / true_l2_norm)

    difference_tree = jax.tree.map(lambda x, y: x - y, treemap_grad, manual_grad)
    l2_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), difference_tree)
    total_l2_norm = jax.tree.reduce(lambda x, y: x + y, l2_norms)
    print(difference_tree['params']['layers_0'], end='\n\n')
    print(total_l2_norm / true_l2_norm)