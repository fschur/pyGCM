import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyGCM",
    version="0.0.1",
    author="Felix Schur",
    author_email="felix.m.schur@gmail.com",
    description="pyGCM (Generalized Covariance Measure)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'pyGCM': 'pyGCM'},
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'GPy',
        'pytest',
        'rpy2',
    ],
)
