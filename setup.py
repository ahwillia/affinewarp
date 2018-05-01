import setuptools

setuptools.setup(
    name="affinewarp",
    version="0.1.0",
    url="https://github.com/ahwillia/affinewarp",

    author="Alex Williams",
    author_email="ahwillia@stanford.edu",

    description="Time warping under affine warping functions",
    # long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
