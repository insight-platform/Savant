from skbuild import setup

setup(
    name='savantboost',
    version='0.0.1',
    description='Python bindings for Savant boost library',
    author='Nikolay Bogoslovskiy',
    author_email='bogoslovskiy_nn@bw-sw.com',
    packages=['pysavantboost'],
    package_dir={'': ''},
    cmake_install_dir='pysavantboost',
    python_requires='>=3.6',
)
