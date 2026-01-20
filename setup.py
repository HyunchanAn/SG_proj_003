from setuptools import setup, find_packages

setup(
    name="vsams",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "numpy",
        "pandas",
        "torch",
        "torchvision"
        # Add other dependencies from requirements.txt if needed
    ],
    author="V-SAMS Team",
    description="Visual-based Surface Analysis & Matching System",
)
