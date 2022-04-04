# janelia_core


**janelia_core** is a Python library containing a core set of machine learning, statistical and dataset tools
originally developed to support Will Bishop's various projects and collaborations at the Janelia Research campus.

These tools are made available here for collaborators as well as any others in the research community who may find
them helpful.


See the [documentation](https://wbishopjanelia.github.io/janelia_core/) for a full API, but here are some of the tools
provided:

1. Objects for representing [time series datasets](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/dataprocessing/dataset/index.html), particularly those derived from imaging datasets.

2. Various machine learning tools, including:

    1. [Conditional distributions](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/ml/torch_distributions/index.html) 
    for use with PyTorch designed specifically for handling structured data.
    2. Penalizers, again designed for use with PyTorch, for flexibly penalizing
    [distributions](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/ml/torch_distributions/index.html) 
    and [PyTorch module parameters](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/ml/torch_parameter_penalizers/index.html).
    3. Tools for representing and fitting general, 
    [non-linear extensions of reduced-rank regression models](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/ml/reduced_rank_models/index.html).
    4. [Custom PyTorch modules](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/ml/extra_torch_modules/index.html), 
    including an optimized function representing a sum of hypercube basis functions over low-dimensional spaces.

3.  Various statistical tools, including those designed for performing statistical inference in various ways with [linear regression models under non-standard noise assumptions](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/stats/regression/index.html).

4.  Various visualization tools, including those for generating [custom colormaps](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/visualization/custom_color_maps/index.html), 
[generating projections of volumetric data](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/visualization/image_generation/index.html), 
and [GUI tools for visualizing data obtained in imaging experiments](https://wbishopjanelia.github.io/janelia_core/autoapi/janelia_core/visualization/exp_viewing/index.html).

## Installation

See the [documentation](https://wbishopjanelia.github.io/janelia_core/install.html) for installation instructions. 

## Contact
bishopw@janelia.hhmi.org  

 

