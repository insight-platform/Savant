from skbuild import setup

setup(
    name='pynvbufsurfacegenerator',
    version='0.0.1',
    description='Python bindings for NvBufSurfaceGenerator library',
    author='Pavel Tomskikh',
    author_email='tomskih_pa@bw-sw.com',
    packages=['pynvbufsurfacegenerator'],
    package_dir={'': ''},
    cmake_install_dir='pynvbufsurfacegenerator',
    python_requires='>=3.6',
)
