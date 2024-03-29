import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='A17',
    version='0.0.2',
    author='Luca Calcaterra',
    description='AUTOML model for Regression and Classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lcalcaterra/A17-MLstarter',
    license='MIT',
    packages=['A17'],
    install_requires=[
        'numpy==1.21.6', 
        'pandas==1.3.5', 
        'sklearn==1.0.2', 
        'xgboost==1.6.1', 
        'optuna==3.0.5'],
)
