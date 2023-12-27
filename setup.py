from setuptools import setup, find_packages


def read_file(file):
    with open(file, "rt") as f:
        return f.read()


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='Image2InChI',
    version='2.0.3',
    description='Convert a picture to InChI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='PanJiaHeng',
    author_email='2905231006@qq.com',
    install_requires=[i for i in read_file("requirements.txt").strip().splitlines() if i != ''],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    license="apache 3.0"
)
