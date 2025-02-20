from setuptools import setup, find_packages

setup(
    name="matrixlib",  # Package name
    version="0.1.0",   # Initial version
    author="Javad Soltanov",
    author_email="javadsoltanov@gmail.com",
    description="A simple matrix library",
    long_description=open("README.md", encoding="utf-8").read(),  # Fix encoding issue
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/matrixlib",  # Change this
    packages=find_packages(),
    install_requires=[],  # Dependencies (empty for now)
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)
