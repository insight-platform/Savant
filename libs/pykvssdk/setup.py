from skbuild import setup

setup(
    name='pykvssdk',
    version='0.0.2',
    description='Python bindings for KVS-SDK library',
    author='Pavel Tomskikh',
    author_email='tomskih_pa@bw-sw.com',
    packages=['pykvssdk'],
    cmake_install_dir='pykvssdk',
    install_requires=[],
    python_requires='>=3.6',
)
