from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="information-sphere",
    version="1.0.0",
    author="北京求一数生科技中心",
    author_email="contact@qiuyishusheng.com",
    description="Information-Oriented Sphere System: 基于信息元的完全可解释深度学习系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiuyishusheng/information-sphere",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black>=22.0", "flake8>=4.0"],
        "experiments": ["torchvision>=0.15.0", "pillow>=9.0.0"],
    },
)

