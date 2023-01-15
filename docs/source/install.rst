Setting up the core library
---------------------------

To install the janelia_core library, follow these steps:

1. Download the code available `here <https://github.com/wbishopJanelia/janelia_core>`_.

2. Navigate to the top-level folder which includes the setup.py script, and

    * To install the code for development purposes run::

        python setup.py develop

    * To install the code for normal use (non-development purposes) run::

        python setup.py install

Dependencies
------------

Most dependencies will be installed by the setup script referenced above. However, there are two additional
dependencies that must be installed separately:

1. PyTorch must be installed, following directions `here <https://pytorch.org/>`_.

2. If you will be working with `klb files <https://bitbucket.org/fernandoamat/keller-lab-block-filetype>`_, you will need to install pyklb which is available from https://github.com/bhoeckendorf/pyklb. Follow the instructions there to install it.  If you are not working with klb files, you can skip this.

3. If you want to modify and generate documentation, you will need sphinx.  Run the following commands to install the required packages:

	* pip install sphinx
	* pip install sphinx-autoapi
	* pip install sphinx-rtd-theme



