from setuptools import find_packages, setup

setup(
    name="diffusion_3d",
    version="0.1",
    package_dir={"": "src/"},
    author="Arjun Agarwal",
    author_email="arjun.agarwal@qure.ai",
    packages=find_packages(),
)
