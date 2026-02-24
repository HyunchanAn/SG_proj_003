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
        "torchvision",
        "timm",
        "albumentations",
        "fastapi",
        "uvicorn",
        "pillow",
        "scikit-learn",
        "requests",
        "tqdm",
        "huggingface_hub"
    ],
    author="V-SAMS Team",
    description="Visual-based Surface Analysis & Matching System",
)
