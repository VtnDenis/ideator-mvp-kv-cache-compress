from setuptools import setup, find_packages

setup(
    name="kv-cache-compress",
    version="0.1.0",
    description="eOptShrinkQ KV cache compression for Hugging Face transformers",
    author="KV-Cache-Compress Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "numpy>=1.24",
    ],
    extras_require={
        "demo": ["datasets", "tqdm"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
