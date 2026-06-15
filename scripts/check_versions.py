import pkg_resources

packages = [
    "vsams",
    "torch",
    "torchvision",
    "timm",
    "streamlit",
    "albumentations",
    "fastapi",
    "uvicorn",
    "numpy",
    "pandas",
    "Pillow",
    "scikit-learn",
    "requests",
    "tqdm",
    "huggingface-hub",
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
