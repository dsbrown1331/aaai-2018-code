#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

//Given one demonstration, find MAP reward and corresponding policy,
//then run policy improvement algorithm by making local changes (changing one action at one state) and calculating the VaR 

using namespace std;



int main() 
{

    ////Experiment parameters
    unsigned int numDemo = 1;            //number of demos to give
    unsigned int rolloutLength = 100;          //max length of each demo
    double alpha = 100; //50                    //confidence param for BIRL
    const unsigned int chain_length = 10000;//1000;//5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    double step = 0.01; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool removeDuplicates = true;
    bool stochastic = false;

    int startSeed = 1321;
    double eps = 0.001;
    
    //test arrays to get features
    const int numFeatures = 2; //white, red, blue, green
    const int numStates = 25;
    const int width = 5;
    const int height = 5;
    double gamma = 0.95;

    vector<unsigned int> initStates = {5,4};
    vector<unsigned int> termStates = {12};
    
    double VaR = 0.99;

//go until can't improve (found local optima)
srand(startSeed);

//create random world //TODO delete it when done
double** stateFeatures = improvementMazeWorld(width, height, numFeatures, numStates);

vector<pair<unsigned int,unsigned int> > good_demos;
vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts

///  create a random weight vector with seed and increment of rep number so same across reps
double* featureWeights = new double[5];
featureWeights[0] = -0.1;      //white
featureWeights[1] = -0.9;   //red
//featureWeights[2] = 0.;
    
FeatureGridMDP fmdp(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
delete[] featureWeights;
//cout << "Transition function" << endl;
//fmdp.displayTransitions();
//cout << "-- Reward function --" << endl;
//fmdp.displayRewards();
///  solve mdp for weights and get optimal policyLoss
vector<unsigned int> opt_policy (fmdp.getNumStates());
fmdp.valueIteration(eps);
//cout << "-- value function ==" << endl;
//fmdp.displayValues();
cout << "features" << endl;
displayStateColorFeatures(stateFeatures, width, height, numFeatures);
//fmdp.deterministicPolicyIteration(opt_policy);
fmdp.calculateQValues();
fmdp.getOptimalPolicy(opt_policy);
cout << "-- optimal policy --" << endl;
fmdp.displayPolicy(opt_policy);

cout << "-- feature weights --" << endl;
fmdp.displayFeatureWeights();

///  generate numDemo demos from the initial state distribution
trajectories.clear(); //used for feature counts
for(unsigned int d = 0; d < numDemo; d++)
{
   int demo_idx = d % initStates.size();
   unsigned int s0 = initStates[demo_idx];
   cout << "demo from " << s0 << endl;
   vector<pair<unsigned int, unsigned int>> traj = fmdp.monte_carlo_argmax_rollout(s0, rolloutLength);
   cout << "trajectory " << d << endl;
   for(pair<unsigned int, unsigned int> p : traj)
       cout << "(" <<  p.first << "," << p.second << ")" << endl;
   trajectories.push_back(traj);
}
//put trajectories into one big vector for birl_test
//weed out duplicate demonstrations
good_demos.clear();
for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
{
    for(pair<unsigned int, unsigned int> p : traj)
        if(removeDuplicates)
        {
            if(std::find(good_demos.begin(), good_demos.end(), p) == good_demos.end())
                good_demos.push_back(p);
        }
        else
        {    
            //Remove terminal states from demos for BIRL
            if(!fmdp.isTerminalState(p.first)) 
                good_demos.push_back(p);
        }
}


///  run BIRL to get chain and Map policyLoss ///
//give it a copy of mdp to initialize
FeatureBIRL birl(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
birl.addPositiveDemos(good_demos);
//birl.displayDemos();
birl.run(eps);
FeatureGridMDP* mapMDP = birl.getMAPmdp();
mapMDP->displayFeatureWeights();
cout << "Recovered reward" << endl;
mapMDP->displayRewards();


//initialize policy to be optimal for Maximum Likelihood reward (MAP under uniform prior)
vector<unsigned int> eval_pi (mapMDP->getNumStates()); 
mapMDP->valueIteration(eps);  
mapMDP->calculateQValues();         
mapMDP->getOptimalPolicy(eval_pi);
cout << "Optimal policy for Maximum Likelihood reward" << endl;
mapMDP->displayPolicy(eval_pi);
//do inline and cheat...
vector<double> evds;

for(unsigned int i=0; i<chain_length; i++)
{
    //cout.precision(5);
    //get sampleMDP from chain
    GridMDP* sampleMDP = (*(birl.getRewardChain() + i));
    //((FeatureGridMDP*)sampleMDP)->displayFeatureWeights();
    //cout << "===================" << endl;
    //cout << "Reward " << i << endl;
    //sampleMDP->displayRewards();
    //cout << "--------" << endl;
    //cout << birl.calculateMaxEntPosterior((FeatureGridMDP*)sampleMDP) << endl;
    vector<unsigned int> sample_pi(sampleMDP->getNumStates());
    //cout << "sample opt policy" << endl;
    sampleMDP->getOptimalPolicy(sample_pi);
    //sampleMDP->displayPolicy(sample_pi);
    //cout << "Value" << endl;
    //sampleMDP->displayValues();
    double Vstar = getExpectedReturn(sampleMDP);
    //cout << "True Exp Val" << endl;
    //cout << Vstar << endl;
    //cout << "Eval Policy" << endl; 
    double Vhat = evaluateExpectedReturn(eval_pi, sampleMDP, eps);
    //cout << Vhat << endl;
    double VabsDiff = abs(Vstar - Vhat);
    evds.push_back(VabsDiff);
    //cout << "abs diff: " << VabsDiff << endl;
    //outfile << VabsDiff << endl;
}    

std::sort (evds.begin(), evds.end());
int VaR_index = (int) chain_length * VaR;
cout << "VaR = " << evds[VaR_index] << endl;
double eval_VaR = evds[VaR_index];
vector<unsigned int> pi_best;
double bestBound;
while(true)
{
    bool foundImprovement = false;
    bestBound = 10000;
    for(int s = 0; s < numStates; s++)
    {
        cout << "step = " << s << endl;
        vector<unsigned int> new_pi = eval_pi;  
        int rand_state = s;          
        //tweak numStates actions
        for(unsigned int i = 0; i < 4; i++)
        {
              new_pi[rand_state] = i;
        

        vector<double> evds_new;
        
        for(unsigned int i=0; i<chain_length; i++)
        {
            //cout.precision(5);
            //get sampleMDP from chain
            GridMDP* sampleMDP = (*(birl.getRewardChain() + i));
            //((FeatureGridMDP*)sampleMDP)->displayFeatureWeights();
            //cout << "===================" << endl;
            //cout << "Reward " << i << endl;
            //sampleMDP->displayRewards();
            //cout << "--------" << endl;
            //cout << birl.calculateMaxEntPosterior((FeatureGridMDP*)sampleMDP) << endl;
            vector<unsigned int> sample_pi(sampleMDP->getNumStates());
            //cout << "sample opt policy" << endl;
            sampleMDP->getOptimalPolicy(sample_pi);
            //sampleMDP->displayPolicy(sample_pi);
            //cout << "Value" << endl;
            //sampleMDP->displayValues();
            double Vstar = getExpectedReturn(sampleMDP);
            //cout << "True Exp Val" << endl;
            //cout << Vstar << endl;
            //cout << "Eval Policy" << endl; 
            double Vhat = evaluateExpectedReturn(new_pi, sampleMDP, eps);
            //cout << Vhat << endl;
            double VDiff = Vstar - Vhat;
            evds_new.push_back(VDiff);
            //cout << "abs diff: " << VabsDiff << endl;
            //outfile << VabsDiff << endl;
        }    
        
        std::sort (evds_new.begin(), evds_new.end());
        int VaR_index = (int) chain_length * VaR;
        //cout << "VaR = " << evds_new[VaR_index] << endl;
        double VaR_new = evds_new[VaR_index];
        //cout << "eval Var = " << eval_VaR << endl;
        //check if best so far
        if(VaR_new < eval_VaR)
        {
            
            foundImprovement = true;
            if(VaR_new < bestBound)
            {
                //cout << "found improvement" << endl;
                bestBound = VaR_new;
                pi_best = new_pi;
                //mapMDP->displayPolicy(pi_best);
            }
        
        }

       }

    }
    if(!foundImprovement)
        break;
    eval_pi = pi_best;
    eval_VaR = bestBound;
    cout << "best" << endl;
    mapMDP->displayPolicy(eval_pi);
    cout << eval_VaR << endl;
}


cout << "minimum VaR policy" << endl;
mapMDP->displayPolicy(eval_pi);
cout << "minimum VaR = " << eval_VaR << endl; 

 //delete world
    for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        delete[] stateFeatures[s1];
    }
    delete[] stateFeatures;



}


