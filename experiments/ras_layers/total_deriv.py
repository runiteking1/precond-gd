import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

# For viz
# import treescope
# treescope.basic_interactive_setup(
#     autovisualize_arrays=True,
#     abbreviation_threshold=2, 
# )
# treescope.register_as_default()

class MLP(nn.Module):
    """
    We first deal with something that's not exactly MLP, but close enough
    """
    num_units: int
    
    def setup(self):
        self.dense1 = nn.Dense(self.num_units)
        self.dense2 = nn.Dense(self.num_units)
    
    def __call__(self, x):
        f = self.dense1(x)
        f = nn.leaky_relu(f)
        f = self.dense2(f)
        # f = nn.tanh(f)
        # x = self.dense2(x)
        return x + f
    

class SimpleMLP(nn.Module):
    num_layers: int
    num_units: int
    num_classes: int

    def setup(self):
        # Create a list of Dense layers
        self.layers = [
            MLP(self.num_units, name=f"layer_{i:02}") for i in range(self.num_layers)
        ]
        # self.classification_layer = nn.Dense(self.num_classes)
        
    def __call__(self, x):
        # Store input to match the notes
        self.sow('intermediates',  f'layer_{0}_output', x)
        
        # Pass the input through each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Store input to match the notes
            self.sow('intermediates', f'layer_{i+1}_output', x)
            
        # Final layer to produce output
        # x = self.classification_layer(x)
        return x

def partition_with_overlap(L, N):
    """
    ChatGPT'd because I'm bad at LeetCode
    Partition the range [0, ..., L] into N overlapping subranges.
    
    Parameters:
        L (int): The maximum integer in the range.
        N (int): The number of partitions.
    
    Returns:
        List[int]: The partition points [c_0, c_1, ..., c_N].
    """
    # Compute the approximate size of each partition
    step = L / N
    
    # Generate the partition points
    c = [round(i * step) for i in range(N + 1)]
    
    return c

def make_matrices(L, ND): 
    """
    Given L layers and ND blocks, give the restriction, pou and q matrices for testing
    """
    restrictions = []
    pous = []
    qs = []

    cs = partition_with_overlap(L, ND)

    for d in range(ND): 
        time_steps = cs[d + 1] - cs[d] + 1
        r = np.zeros((time_steps, L + 1))
        for t in range(time_steps):
            r[t, cs[d] + t] = 1

        pou = np.eye(time_steps)
        if d == 0:
            pou[-1, -1] = 0.5
        elif d == ND - 1:
            pou[0, 0] = 0.5
        else: 
            pou[0, 0] = 0.5
            pou[-1, -1] = 0.5

        if d == 0:
            q = np.zeros((time_steps, L + 1))
            for i in range(time_steps): 
                q[i, i] = 1
        else:
            q = np.zeros((time_steps + 1, L + 1))
            for t in range(time_steps + 1): 
                q[t, cs[d] - 1 + t] = 1
        restrictions.append(r)
        pous.append(pou)
        qs.append(q)

    # assert np.linalg.norm(R1.T @ D1 @ R1 + R2.T @ D2 @ R2 + R3.T @ D3 @ R3 - np.eye(10)) < 1e-10

    return restrictions, pous, qs


if __name__ == "__main__":
    n = 1 # 1 dimensional input and output 
    L = 10 # Number of layers; trying to emulate the paper

    # Fake data
    n_samples = 1
    x = jnp.ones((n_samples, n))
    # y = jnp.ones((n_samples, n)) * .1
    
    # First, pretend to call the model
    model = SimpleMLP(num_layers=L, num_units=n, num_classes=n)
    params = model.init( jax.random.PRNGKey(0), x)
    predictions, intermediates = model.apply(params, x, mutable=['intermediates'])
    # Create layer model; then be able to take the gradients and stuff
    layer = MLP(num_units=n)
    # layer.apply({'params': params['params']['layers_0']}, x)
    
    # First define function; one input
    def apply_layer(params, x): 
        assert len(x) == 1
        return jnp.squeeze(layer.apply(params, x))
    
    # The issue with jax.grad is it doesn't automatically vectorize over batch dimension, so we do it manually with vmap and squeeze
    K_i_one = jax.grad(apply_layer, 0)
    K_i = jax.vmap(K_i_one, (None, 0)) # We don't allow vmapping over the params. I guess we could in theory... but it's easier to wrap my head around htis
    
    M_i_one = jax.grad(apply_layer, 1)
    M_i = jax.vmap(M_i_one, (None, 0))

    dgdu = np.eye(L + 1) # It's tridiagonal with ones down diagonal; will have to change once scaled up
    for l in range(L):
        dgdu[l, l+1] =  -jnp.squeeze(M_i({'params': params['params'][f'layer_{l:02}']}, intermediates['intermediates'][f'layer_{l}_output'][0]))

    # Number of trainable parameters
    p = sum(x.size for x in jax.tree.leaves(params))
    
    dgdt = np.zeros((p, L + 1))
    # This is harder to construct; need a mapping from dof to matrix... think about this for bigger problems
    # Better way would be a matrix free operation; but for now, just do 
    counter = 0
    for l in range(L): 
        flattened, _ = jax.tree.flatten(
                K_i({'params': params['params'][f'layer_{l:02}']}, intermediates['intermediates'][f'layer_{l}_output'][0])
        )
        vals = np.array([jnp.squeeze(x) for x in flattened])
        layer_p = len(vals)
    
        # Can make this dynamic 
        dgdt[counter:counter + layer_p, l + 1] = vals

        counter += layer_p

    # Total derivative to calculate Jacobian 
    derivative = np.linalg.inv(dgdu.T) @ dgdt.T


    # Let's compare this with the Jacobian 
    jacobian = jax.jacobian(model.apply, argnums=0)(params, x)
    jacobian = jnp.array([jnp.squeeze(x) for x in jax.tree.flatten(jacobian)[0]])

    print('Relative norm of diff: ', jnp.linalg.norm(jacobian - derivative[-1, :]) / jnp.linalg.norm(jacobian))
    
    ###############
    # ASM
    #########

    ND = 4
    
    print(partition_with_overlap(L, ND)) 

    
    restrictions, pous, qs = make_matrices(L, ND)
    # Construct RAS preconditioner 
    ras = np.zeros((L + 1, L + 1))
    for i in range(ND): 
        R = restrictions[i]
        D = pous[i]
        ras += R.T @ D @ jnp.linalg.inv(R @ dgdu.T @ dgdu @ R.T) @ R
    
    # Technically, this should be SPD, but then need to do the whole P^{1/2} business. 
    print(np.linalg.eigvals(ras @  dgdu.T @ dgdu))

    rasq = np.zeros((L + 1, L + 1))
    for i in range(ND): 
        R = restrictions[i]
        D = pous[i]
        Q = qs[i]
        Js = np.linalg.inv(Q @ dgdu @ Q.T)
        rasq += R.T @ D @ (R @ Q.T @ Js @ Js.T @ Q @ R.T) @ R
    
    # Technically, this should be SPD, but then need to do the whole P^{1/2} business. 
    print(jnp.linalg.eigvals(rasq @  dgdu.T @ dgdu))

    z = np.zeros((L+1, ND))
    for i in range(ND): 
        ones = np.ones((L+1,))
        z_i = restrictions[i].T @ pous[i] @ restrictions[i] @ ones
        z[:, i] = z_i
    R0 = z.T

    rasq_wc = R0.T @ np.linalg.inv(R0 @ dgdu.T @ dgdu @ R0.T) @ R0 + rasq
    # rasq_wc = rasq @ (R0.T @ np.linalg.inv(R0 @ dgdu.T @ dgdu @ R0.T) @ R0)
    print(
        np.sort(np.linalg.eigvals((R0.T @ np.linalg.inv(R0 @ dgdu.T @ dgdu @ R0.T) @ R0) @  dgdu.T @ dgdu))
    )

    print(
        np.sort(np.linalg.eigvals((rasq_wc) @  dgdu.T @ dgdu))
    )
