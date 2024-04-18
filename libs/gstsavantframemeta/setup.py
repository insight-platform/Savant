from skbuild import setup

setup(
    name='pygstsavantframemeta',
    version='0.0.3',
    description='Python bindings for GstSavantFrameMeta library',
    author='Pavel Tomskikh',
    author_email='tomskih_pa@bw-sw.com',
    packages=['pygstsavantframemeta'],
    package_dir={'': ''},
    cmake_install_dir='pygstsavantframemeta',
    python_requires='>=3.6',
)
