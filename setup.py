from setuptools import setup, find_packages

setup(
    name="adaptive_learner_v0",
    version="1.0.0",
    description="Adaptive learning path optimization for personalized education",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "networkx>=2.6.0",
    ],
)
