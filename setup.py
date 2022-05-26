from setuptools import setup, find_packages
import os

cur_path = os.path.abspath(os.path.dirname(__file__))

with open(cur_path + "/requirements.txt", encoding="utf-8") as fp:
    requirements = [rq.rstrip() for rq in fp.readlines() if not rq.startswith("#")]

setup(
    name="botools",
    packages=find_packages(),
    license="MIT",
    version="0.1.0",  # 版本
    install_requires=requirements,
)
