"""
Setup script for CGL-DV.
"""

from setuptools import setup, find_packages

with open( "README.md", "r", encoding = "utf-8" ) as fh:
    long_description = fh.read()

with open( "requirements.txt", "r", encoding = "utf-8" ) as fh:
    requirements = [ line.strip() for line in fh if line.strip() and not line.startswith( "#" ) ]

setup(
        name = "cgldv",
        version = "1.0.0",
        author = "Xuan Zeng, Cui Zhu, Wenjun Zhu",
        author_email = "cuizhu@bjut.edu.cn",
        description = "Confidence-Guided Dual-View Learning on Text-Attributed Graphs with LLMs",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        url = "https://github.com/yourusername/CGL-DV",
        packages = find_packages(),
        classifiers = [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires = ">=3.8",
        install_requires = requirements,
        extras_require = {
            "dev": [
                "pytest>=7.0.0",
                "black>=22.0.0",
                "flake8>=4.0.0",
                "mypy>=0.950",
            ],
        },
)
