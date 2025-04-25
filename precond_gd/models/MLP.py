from flax import linen as nn
from flax.linen import initializers


# Create standard model
class MLP_1D(nn.Module):
    num_layers: int = 2
    hidden_dim: int = 6

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, 1)
        for _ in range(self.num_layers):
            x = nn.Dense(
                features=self.hidden_dim, use_bias=True,
                kernel_init=initializers.kaiming_uniform(),  # Kaiming uniform initialization
                bias_init=initializers.zeros  # Bias initialized to zero
            )(x)
            x = nn.tanh(x)  # Apply activation function after each layer
        x = nn.Dense(features=1)(x)
        return x
