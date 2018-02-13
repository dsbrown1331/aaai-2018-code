# Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning
## Daniel S. Brown and Scott Niekum
### Follow the instructions below to reproduce results in our <a href="https://arxiv.org/abs/1707.00724">AAAI 2018</a> and <a href="https://www.cs.utexas.edu/~dsbrown/pubs/Brown_AAAIFS17.pdf">AAAI 2017 Fall Symposium</a> papers.

  - UNDER CONSTRUCTION
  
  #### Dependencies
  - Matplotlib (for generating figures)
  - Python3 (for running scripts)
  
  #### Getting started
  - Make a build directory: `mkdir build`
  - Make a data directory to hold results: `mkdir data`
  
  #### Infinite Horizon GridWorld (Figure 2 in <a href="https://arxiv.org/abs/1707.00724">AAAI 2018 paper</a>)
  - Use `make gridworld_basic_exp` to build the experiment.
  - Execute `./gridworld_basic_exp` to run. Data will be output to `./data/gridworld`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldBasicExperiment.cpp`. 
  -Once experiment has finished run `python scripts/generateGridWorldBasicPlots.py` to generate figures used in paper.
  
  
  
  #### Sensitivity to Confidence Parameter (Figure 3 in <a href="https://arxiv.org/abs/1707.00724">AAAI 2018 paper</a>)
  - Use `make gridworld_noisydemo_exp` to build the experiment.
  - Execute `./gridworld_noisydemo_exp` to run. Data will be output to `./data/gridworld_noisydemo_exp/`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldNoisyDemoExperiment.cpp`. 
  -Once experiment has finished run `python scripts/generateNoisyDemoPlots.py` to generate figures used in paper.
  
  
  #### Comparison with theoretical bounds (Table 1 in in <a href="https://arxiv.org/abs/1707.00724">AAAI 2018 paper</a>)
  - Use `make gridworld_projection_exp` to build the experiment.
  - Execute `./gridworld_projection_exp` to run. Data will be output to `./data/abbeel_projection/`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldProjectionEvalExperiment.cpp`. 
  -Once experiment has finished run `python scripts/generateProjectionEvalTable.py` to generate table used in paper.
  

