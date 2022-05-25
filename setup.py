from setuptools import setup, find_packages

setup(
    name='botools',
    packages=find_packages(),
    license='MIT',
    version='0.1.0',    # 版本
    install_requires=[  # 依赖
        'tqdm',              # 进度条
        'opencv-python',
        'numpy',
    ]
)