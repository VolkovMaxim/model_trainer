from setuptools import find_packages, setup


setup(name="model_trainer",
      packages=find_packages(include=["model_trainer"]),
      vesion="0.1.0",
      description="True validation",
      author="wolk1612",
      install_requires=["tqdm", "IPython", "torch", "numpy", "scipy", "matplotlib"]
      )
