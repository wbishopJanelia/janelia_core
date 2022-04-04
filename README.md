# janelia_core
Core code for William Bishop's work at Janelia.

## Dependencies

1) conda
2) pyklb
3) pytorch

## Setting up

1) Create a conda environment by running this command:  
    - conda create -n janelia_core python=3.7

2) Activate the conda environment by running this command:
    - Mac: source activate janelia_core
    - Windows: conda activate janelia_core
3) In the directory containing setup.py run the command from the terminal: 
	- python setup.py develop

4) Install jupyter notebook by running this command
	conda install jupyter

5) Install pytorch by running this command:
    - Mac: conda install pytorch torchvision -c pytorch
    - Windows: conda install pytorch -c pytorch, pip install torchvision
    
6) Optionally install moviepy by running this command:  pip install moviepy==2.0.0.dev1.
    - moviepy is required for only one visualization function.  If you do not need this function, there is 
    no need to install it - the rest of the code will function fine without it

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

##### Note on installing pyklb for Windows

There is an bug in the current version of pyklb so that after following the instructions referenced 
above the klb.dll library is not in the right spot.  To fix this, find the klb.dll file in the build/lib folder 
under the pyklb folder of the pyklb project and copy it to the site packages folder for your environment.
 

