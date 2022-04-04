.. We use autoapi to generate api documentation: See https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html

Welcome to janelia_core's documentation!
========================================

**janelia_core** is a Python library containing a core set of machine learning, statistical and dataset tools
originally developed to support Will Bishop's various projects and collaborations at the Janelia Research campus.

These tools are made available here for collaborators as well as any others in the research community who may find
them helpful.

.. note:: This project is under active development.

Provided Tools
--------------

See the full :doc:`API <autoapi/janelia_core/index>` to explore all tools which are provided in the library.  Some highlights:

1. Objects for representing :doc:`time series datasets <autoapi/janelia_core/dataprocessing/dataset/index>`, particularly those derived from imaging datasets.

2. Various machine learning tools, including:

    1. :doc:`Conditional distributions<autoapi/janelia_core/ml/torch_distributions/index>` for use with PyTorch designed specifically for handling structured data.
    2. Penalizers, again designed for use with PyTorch, for flexibly penalizing :doc:`distributions<autoapi/janelia_core/ml/torch_distributions/index>` and :doc:`PyTorch module parameters<autoapi/janelia_core/ml/torch_parameter_penalizers/index>`.
    3. Tools for representing and fitting general, :doc:`non-linear extensions of reduced-rank regression models<autoapi/janelia_core/ml/reduced_rank_models/index>`.
    4. :doc:`Custom PyTorch modules <autoapi/janelia_core/ml/extra_torch_modules/index>`, including an optimized function of hypercube basis functions over low-dimensional spaces.

3.  Various statistical tools, including those designed for performing statistical inference in various ways with :doc:`linear regression models under non-standard noise assumptions<autoapi/janelia_core/stats/regression/index>`.

4.  Various visualization tools, including those for generating :doc:`custom colormaps<autoapi/janelia_core/visualization/custom_color_maps/index>`, :doc:`generating projections of volumetric data<autoapi/janelia_core/visualization/image_generation/index>`, and :doc:`GUI tools for visualizing data obtained in imaging experiments <autoapi/janelia_core/visualization/exp_viewing/index>`.


.. toctree::
   :maxdepth: 2
   :caption: Installation

   Installation and Getting Started <install>

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   janelia_core <autoapi/janelia_core/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
