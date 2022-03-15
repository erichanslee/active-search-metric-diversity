# WinterSim Repo

# Installation
Run `python setup.py install` in the command line. 

# Directory Structure
The `src/` directory is subdivided as follows:

* `src/acquisitions/` contains implementations of all acquisition functions. 
* `src/model/` contains our GP implementation. By default, this implementation uses a constant mean function and the Matern 5/2 kernel. 
* `src/runners/` contains the BO runners, which are invoked to run a full BO loop.
* `src/test_problems/` contains implementations of synthetic functions, which are used to benchmark BO acquisition functions. 

# Demos 
We have a few simple demos in the `demos/` folder, which you should run to get started. 
