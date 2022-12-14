from skbuild import setup

version_file = 'VERSION'
with open(version_file, 'r') as file_obj:
    content = file_obj.read().splitlines()
    version = content[0]

setup(
    name='savantboost',
    version=version,
    description='Python binding for Savant boost library',
    author='Nikolay Bogoslovskiy',
    author_email='bogoslovskiy_nn@bw-sw.com',
    license='Apache License 2.0',
    packages=['pysavantboost'],
    package_dir={'': ''},
    cmake_install_dir='pysavantboost',
    data_files=[('./', [version_file])],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.6',
)
