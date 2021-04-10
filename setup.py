from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Alpha',
    'Intended Audience :: Education',
    'Operating System :: Linux :: Ubuntu :: Ubuntu 20.',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='modani2x',
    version='0.0.1',
    description='A simple basic script to test package publications.',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',  
    author='Takshan',
    author_email='takshan@naver.com',
    license='MIT', 
    classifiers=classifiers,
    keywords='modani2x', 
    packages=find_packages(),
    install_requires=['torch', 'numpy','torchani'] 
)