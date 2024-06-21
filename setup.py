import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nerval",
    version="1.1.2",
    author="Mariangela D'Addato",
    author_email="mdadda.py@gmail.com",
    description="Entity-level confusion matrix and classification report to evaluate Named Entity Recognition (NER) models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maridda/nerval",
    project_urls={
        "Bug Tracker": "https://github.com/maridda/nerval/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=("tests")),
    python_requires=">=3.6",
)
