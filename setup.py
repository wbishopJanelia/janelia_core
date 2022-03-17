from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='janelia_core',
    version='0.1.0',
    author='William Bishop',
    author_email='bishopw@hhmi.org',
    packages=['janelia_core'],
    python_requires='>=3.7.0',
    description='Core code which supports William Bishops research projects at Janelia.',
    long_description = long_description,
    install_requires=[
    "dipy",
	"findspark",
	"h5py",
    "imageio-ffmpeg",
	"jupyter",
	"matplotlib",
    "morphsnakes",
    "numpy",
	"pyqtgraph",
	"scikit-image",
	"scikit-learn",
    "pandas",
    "psutil",
	"pyspark",
    "sphinx",
    "sphinx-autoapi",
    "sphinx_rtd_theme",
    "tqdm",
    ],
)
