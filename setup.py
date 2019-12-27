"""
side_effects
Predict DDI and their side effects
InVivo AI

"""
import glob

from setuptools import setup, find_packages

import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:]),

setup(
    # Self-descriptive entries which should always be present
    name='side_effects',
    author='InVivo AI',
    author_email='rogia@invivoai.com',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='Not Open Source',
    scripts=glob.glob('bin/*'),

    # Which Python importable modules should be included when your package is installed
    packages=['side_effects'],

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.5",  # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
   # packages=find_packages()
)
