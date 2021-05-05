from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Education',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]



setup(
    name='modani2x',
    version='0.0.5',
    description='A simple basic script to test package publications.',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',  
    author='Takshan',
    author_email='takshan@naver.com',
    license='MIT', 
    classifiers=classifiers,
    keywords='mod_ani2x', 
    packages=find_packages(),
    console_scripts =
        ["modani2x = modani2x.modani2x"],
    scripts=['modani2x'],
    install_requires=['torch', 'numpy','torchani', 'ase', 'torchani'] ,
    project_urls={
        'Source': 'https://github.com/Takshan/modani2x',
    }
)