from setuptools import setup, find_packages
from pathlib import Path
from setuptools import setup

PATH_HERE = Path(__file__).parent

with open(PATH_HERE / "requirements.txt", encoding="utf-8") as fp:
    requirements = [rq.rstrip() for rq in fp.readlines() if not rq.startswith("#")]

setup(
    name="botools",
    packages=find_packages(),
    license="MIT",
    version="0.1.0",  # 版本
    install_requires=requirements,
)
