# janelia_core
Core code for William Bishop's work at Janelia.

## Dependencies

1) conda
2) pyklb
3) python-graphviz

## Setting up for development purposes

### On Mac:

1) Create a conda environment by running this command: 
	conda create -n janelia_core python=3.6

2) Activate the conda environment by running this command
    source activate janelia_core

3) In the directory containing setup.py run the command from the terminal: 
	python setup.py develop

4) Install jupyter notebook by running this command
	conda install jupyter

5) Install python-graphviz with the command: 
    conda install python-graphviz

#### Installing pyklb

If you will be working with [klb files](https://bitbucket.org/fernandoamat/keller-lab-block-filetype), 
you will need to install pyklb which is available from https://github.com/bhoeckendorf/pyklb. 
Follow the instructions there to install into the janelia_core conda environment. 

##### Note on installing pyklb for mac

There is an bug in the current version of pyklb so that after following the instructions referenced 
above you must use Mac's install_name_tool command and run:
 
   install_name_tool -change /Users/clackn/src/keller-lab-block-filetype/build/libklb.dylib [path to downloaded dylib] [path to the .so file in site-packages]

The dylib file will by default be in the build/lib directory in the top level folder of the pyklb repository.  
The .so file will be in the site-packages folder of the conda environment (e.g., /anaconda3/envs/janelia_core/lib/python3.6/site-packages/pyklb.cpython-36m-darwin.so). 

### On Windows:

1) Create a conda environment by running this command: 
	conda create -n janelia_core python=3.6

2) Activate the conda environment by running this command
    source activate janelia_core

3) In the directory containing setup.py run the command from the terminal: 
	python setup.py develop

4) Install jupyter notebook by running this command
	conda install jupyter

5) Install python-graphviz with the command: 
    conda install python-graphviz

