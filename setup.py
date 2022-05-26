from setuptools import setup, find_packages


requirements=['tqdm', 'opencv-python', 'numpy', 'torch']

setup(
    name="botools",
    packages=find_packages(),
    license="MIT",
    version="0.1.0",  # 版本
    install_requires=requirements,
)
