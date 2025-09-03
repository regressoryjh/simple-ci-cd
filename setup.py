from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="simple-ci-cd",
    version="0.1.0",
    author="regressoryjh",
    author_email="your.email@example.com",  # Ganti dengan email Anda
    description="A simple ML CI/CD project with GitHub Actions for house price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/regressoryjh/simple-ci-cd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=5.0",
            "black>=22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-train=src.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)