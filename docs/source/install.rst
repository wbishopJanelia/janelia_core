Setting up the core library
---------------------------

To install the janelia_core library, follow these steps:

1. Navigate to the top-level folder with the setup.py script

2. To install the code for development purposes run::

    python setup.py develop

3. To install the code for normal use (non-development purposes) run::

    python setup.py install

Dependencies
------------

Most dependencies will be installed by the setup script referenced above. However, there are two additional
dependencies that must be installed separately:

1. PyTorch must be installed, following directions `here <https://pytorch.org/>`_.

2. If you will be working with [klb files](https://bitbucket.org/fernandoamat/keller-lab-block-filetype),
you will need to install pyklb which is available from https://github.com/bhoeckendorf/pyklb.
Follow the instructions there to install it.  If you are not working with klb files, you can skip this.




