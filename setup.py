from setuptools import find_packages, setup


# Read contents of readme.md to use as long description later
with open('README.md') as f:
    readme = f.read()


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
    install_requires = ['numpy==1.26.4', 'matplotlib==3.8.4', 'setuptools==69.5.1'],
    extras_require = ['graphviz==0.20.3', 'cupy_cuda12x==13.1.0'],
    keywords = ['deepfusion', 'deep learning', 'neural networks', 'artificial intelligence', 'machine learning',
                'model', 'optimization', 'backpropagation'],
)