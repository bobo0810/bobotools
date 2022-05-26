from setuptools import setup, find_packages


setup(
    name="bobotools",
    packages=find_packages(),
    license="MIT",
    author="bobo0810",
    description="bobotools",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    version="0.1.0",  # 版本
    install_requires=["tqdm", "opencv-python", "numpy", "torch"],
    python_requires=">=3.6",
    include_package_data=True,
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)
