from setuptools import setup, find_packages

setup(
    name="precond_gd",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flax~=0.10.4",
        "jax", #"[cuda]~=0.5.2",   # Or plain "jax" depending on CPU/GPU
        "chex~=0.1.89",
        "matplotlib~=3.9.4",
        "tqdm~=4.67.1",
        "seaborn~=0.13.2",
        "ipykernel",
        "jupyter",
        "notebook",
        "numpy~=2.2.3",
        "tensorflow_datasets",
        "optax~=0.2.4",
    ],
    python_requires=">=3.9",  # Adjust as needed
)

