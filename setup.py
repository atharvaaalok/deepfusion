from setuptools import find_packages, setup


# Read contents of readme.md to use as long description later
with open('README.md') as f:
    readme = f.read()

# this grabs the requirements from requirements.txt, assumes that there are no comments in the file
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]


setup(
    name = 'deepfusion',
    author = 'Atharva Aalok',
    author_email = 'atharvaaalok@gmail.com',
    version = '0.0.0',
    url = 'https://github.com/atharvaaalok/deepfusion',
    license = 'MIT',
    description = 'A highly modular and customizable deep learning framework!',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    packages = find_packages(),
    python_requires = '>=3.12',
    install_requires = REQUIREMENTS,
    keywords = ['deepfusion', 'deep learning', 'neural networks', 'artificial intelligence', 'machine learning',
                'model', 'optimization', 'backpropagation'],
)