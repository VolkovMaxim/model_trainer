from setuptools import find_packages, setup
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_requirements():
    """parses requirements from requirements.txt"""
    reqs_path = os.path.join(__location__, 'requirements.txt')
    with open(reqs_path, encoding='utf8') as f:
        reqs = [line.strip() for line in f if not line.strip().startswith('#')]

    names = []
    links = []
    for req in reqs:
        if '://' in req:
            links.append(req)
        else:
            names.append(req)
    return {'install_requires': names, 'dependency_links': links}


setup(name="model_trainer",
      packages=find_packages(include=["mypythonlib"]),
      vesion="0.1.0",
      description="True validation",
      author="wolk1612",
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      **read_requirements()
      )
