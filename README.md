# Efficient Probabilistic Performance Bounds for Inverse Reinforcement Learning
## Daniel S. Brown and Scott Niekum
### Follow the instructions below to reproduce results in our [AAAI 2018](https://arxiv.org/abs/1707.00724) and our [AAAI 2017 Fall Symposium](https://www.cs.utexas.edu/~dsbrown/pubs/Brown_AAAIFS17.pdf) papers.

 
  #### Dependencies
  - Matplotlib (for generating figures)
  - Python3 (for running scripts)
  
  #### Getting started
  - Make a build directory: `mkdir build`
  - Make a data directory to hold results: `mkdir data`
  
  #### Infinite Horizon GridWorld (Figure 2 in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))
  - Use `make gridworld_basic_exp` to build the experiment.
  - Execute `./gridworld_basic_exp` to run. Data will be output to `./data/gridworld`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldBasicExperiment.cpp`. 
  - Once experiment has finished run `python scripts/generateGridWorldBasicPlots.py` to generate figures used in paper.
  - You should get something similar to the following two plots

<div>
  <img src="figs/boundAccuracy.png" width="350">
  <img src="figs/boundError.png" width="350">
</div>
  
  
  #### Sensitivity to Confidence Parameter (Figure 3 in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))

  - Use `make gridworld_noisydemo_exp` to build the experiment.
  - Execute `./gridworld_noisydemo_exp` to run. Data will be output to `./data/gridworld_noisydemo_exp/`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldNoisyDemoExperiment.cpp`. 
  - Once experiment has finished run `python scripts/generateNoisyDemoPlots.py` to generate figures used in paper.
  - You should get something similar to the following two plots

<div>
  <img src="figs/noisydemo_accuracy_overAlpha.png" width="350">
  <img src="figs/noisydemo_bound_error_overAlpha.png" width="350">
</div>

   - Note that the bounds when c=0 are different than shown in paper. We are working on determining the reason for this discrepancy.
  
  
  #### Comparison with theoretical bounds (Table 1 in in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))
  - Use `make gridworld_projection_exp` to build the experiment.
  - Execute `./gridworld_projection_exp` to run. Data will be output to `./data/abbeel_projection/`
  - Experiment will take some time to run since it runs 200 replicates for each number of demonstrations. Experiment parameters can be set in `src/gridWorldProjectionEvalExperiment.cpp`. 
  - Once experiment has finished run `python scripts/generateProjectionEvalData.py` to generate data used in paper.
  - We reran the experiment from our paper and got the following results (slightly different from paper due to random seeding):
  
   | Bound            | 1 demo | 5 demos | 9 demos | 23052 demos | Ave Accuracy |
| ------------------- |:-----:   | :----:   | :----:    | :----:        | :----:        |
| 0.95-VaR EVD Bound  |  0.9392 | 0.2570 | 0.1370 | - | 0.98|
| 0.99-VaR EVD Bound  |1.1448  | 0.2972  | 0.1575 | - |  1.0 |
| Syed and Schapire 2008  | 142.59 | 63.77  | 47.53   | 0.9392 | 1.0 |
  
  
  #### Policy Selection for Driving Domain
  - UNDER CONSTRUCTION
  
  #### Policy Improvement (Figure 4 in [AAAI 2018 paper](https://arxiv.org/abs/1707.00724))
  - Use `make improvement_exp` to build the experiment.
  - Execute `./improvement_exp` to run. 
  - The minimum VaR policy will be printed to the terminal. 
  
  
  #### Demonstration Sufficiency (Figure 4 in  [AAAI 2017 Fall Symposium Paper](https://www.cs.utexas.edu/~dsbrown/pubs/Brown_AAAIFS17.pdf))
  - Use `make demo_sufficiency_exp` to build the experiment.
  - Execute `./demo_sufficiency_exp` to run. Data will be output to `./data/demo_sufficiency/`
  - Once experiment has finished run `python scripts/generateDemoSufficiencyPlot.py` to generate plot 4 (b).
  - You should get the following figure.
  <div>
  <img src="figs/demoSufficiency.png" width="350">
  </div>
  - Note that given a non-zero safety threshold on Value-at-Risk, say &epsilon; = 0.01, the agent would be able to report that it had learned the given task after two demonstrations, whereas using only feature counts makes it seem like three demonstrations are needed.
  

