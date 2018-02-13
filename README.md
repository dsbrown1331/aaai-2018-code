# Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning
## Daniel S. Brown and Scott Niekum
## The University of Texas at Austin
### Follow the instructions below to reproduce results in our <a href="https://arxiv.org/abs/1707.00724">AAAI 2018</a> and <a href="https://www.cs.utexas.edu/~dsbrown/pubs/Brown_AAAIFS17.pdf">AAAI 2017 Fall Symposium</a> papers.

  - UNDER CONSTRUCTION
  
  #### Getting started
  - Make a build directory: `mkdir build`
  - Make a data directory to hold results: `mkdir data`
  
  #### Infinite Horizon GridWorld
  - Use `make gridworld_basic_exp` to build the experiment for generating Figure 1 in the paper.
  - Execute `./gridworld_basic_exp` to run. Data will be output to `./data/gridworld_basic_exp`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldBasicExperiment.cpp`. 

