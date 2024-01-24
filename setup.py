from setuptools import setup, find_packages

install_requires = [
    "numpy >= 1.24.3",
    "pandas >= 1.5.3",
    "matplotlib >= 3.7.1",
    "seaborn >= 0.12.2",
    "rfcde == 0.3.2",
    "statsmodels >= 0.14.0",
    "scipy >= 1.11.0",
    "scikit-learn == 0.24.2",
    "skgrf == 0.3.0",
    "copulae == 0.7.8"
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='EWS_CondEnt',
      version='1.0.0',
      description='concept drift EWS',
      long_description='Early warning system for online concept drift detection via conditional entropy estimation',
      long_description_content_type='text/markdown',
      url='',
      author='Artyom Kraevskiy, Evgeniy Sokolovskiy, Artyom Prokhorov',
      author_email='akraevskiy@hse.ru, esokolovskii@gmail.com, artem.prokhorov@sydney.edu.au',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.9',
      zip_safe=False,
)