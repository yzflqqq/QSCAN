from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qscan',
    version='0.1.0',
    description='Quality-aware Semantic Conv-Attention Network for image super-resolution',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(exclude=('tests', 'tests.*')),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.5.0',
        'einops>=0.7.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
