from setuptools import find_packages, setup
import codecs

with codecs.open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    "torch>=1.1.0",
    "transformers>=4.0.0",
    "tensorboardX"
]

setup(
    name="deepclassifier",
    version="0.0.5",
    author="Zichao Li",
    author_email="2843656167@qq.com",
    description="DeepClassifier is aimed at building general text classification model library.It's easy and user-friendly to build any text classification task.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codewithzichao/DeepClassifier",
    download_url='https://github.com/codewithzichao/DeepClassifier/tags',
    packages=find_packages(
        exclude=["tests"]
    ),
    python_requires=">=3.6.0",
    install_requires=REQUIRED_PACKAGES,
    #extra_require={},
    entry_points={},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="Apache-2.0",
    keywords=[
        "text classification", "pytorch", "torch", "NLP", "deep learning", "deepclassifier"
    ]

)
