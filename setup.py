from setuptools import find_packages, setup
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


setup(name="model_trainer",
      packages=find_packages(include=["model_trainer"]),
      vesion="0.1.0",
      description="True validation",
      author="wolk1612",
      )