# WinterSim Repo
Code release for the paper [Achieving Diversity in Objective Space for Sample-Efficient Search of Multiobjective Optimization Problems](https://arxiv.org/abs/2306.13780), published in Winter Sim 2022. 
The paper and this code release develop a method similar to active search. The method allows users to input a multiobjective design or optimization problem and obtain a set of diverse solutions. 
The diversity of this solution set is important to a range of scientific and engineering applications. 

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
