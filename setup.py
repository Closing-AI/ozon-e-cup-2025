from setuptools import setup, find_packages

setup(
    name="text_features",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'pandas>=1.5.0',
        'scikit-learn>=1.2.0',
        'razdel>=0.5.0',
        'pymorphy3>=1.2.0'
    ],
)
