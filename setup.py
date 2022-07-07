from setuptools import setup, find_packages
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# f= open (os.path.join(ROOT_DIR, 'bobotools/requirements.txt'),'r')
# install_requires = [line.strip()  for line in  f.readlines()]
# f.close()
install_requires = [
    "pytest",
    "tqdm",
    "opencv-python-headless",
    "numpy",
    "ptflops",
    "torch",
    "torchvision",
    "grad-cam",
    "pillow",
]

setup(
    name="bobotools",
    packages=find_packages(),
    license="MIT",
    author="bobo0810",
    description="bobotools",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    version="0.4.7.5",  # 版本
    install_requires=install_requires,
    python_requires=">=3.6",
    include_package_data=True,
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
