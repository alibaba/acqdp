from setuptools import setup, find_packages
import os.path

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), "r") as fh:
    long_description = fh.read()

zip_safe = True
try:
    from Cython.Build import cythonize
    zip_safe = False
except ImportError:
    print('Installing QDP without Cython...')

    def cythonize(*args, **kwargs):
        return []

setup(
    name='acqdp',
    version='0.1.1',
    description='Alibaba Cloud Quantum Development Platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alibaba Quantum Lab',
    author_email='aql_software@alibabacloud.com',
    license='MIT',
    url='https://github.com/alibaba/acqdp',
    packages=find_packages(include=['acqdp*']),
    package_data={'acqdp': ['*.ini', '*.json']},
    include_package_data=True,
    ext_modules=cythonize('acqdp/utility/*.pyx'),
    zip_safe=zip_safe,
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'numexpr',
        'matplotlib',
        'opt_einsum',
        'kahypar',
        'cma',
        'jax',
        'jaxlib',
        'tqdm'
    ],
    python_requires='>=3.7'
)
