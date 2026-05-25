
import pkg_resources
import torch
import torchvision
import timm
import streamlit
import albumentations
import fastapi
import uvicorn
import numpy
import pandas
import PIL
import sklearn
import requests
import tqdm
import huggingface_hub

packages = [
    "vsams", "torch", "torchvision", "timm", "streamlit", "albumentations", 
    "fastapi", "uvicorn", "numpy", "pandas", "Pillow", "scikit-learn", 
    "requests", "tqdm", "huggingface-hub"
]

print("-" * 30)
for pkg in packages:
    try:
        dist = pkg_resources.get_distribution(pkg)
        print(f"{pkg:20} : {dist.version}")
    except pkg_resources.DistributionNotFound:
        print(f"{pkg:20} : NOT FOUND")
    except Exception as e:
        print(f"{pkg:20} : Error - {e}")
print("-" * 30)
