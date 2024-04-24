from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Efficient bm25 implementation compatible with sklearn'
LONG_DESCRIPTION = 'Efficient bm25 implementation compatible with sklearn'

# Setting up
setup(
        name="bm25", 
        version=VERSION,
        author="Ayoub Hammal",
        author_email="<ayoubhammal@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "scikit-learn"
        ],
        keywords=['python', 'bm25', 'scikit-learn'],
)
