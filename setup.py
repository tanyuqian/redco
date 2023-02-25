from setuptools import find_packages, setup


setup(
    name="redco",
    version="0.0.1",
    author="Bowen Tan",
    packages=find_packages(),
    install_requires=['jax', 'flax', 'optax', 'numpy'],
    include_package_data=True,
    python_requires=">=3.8"
)