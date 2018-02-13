#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

///test out whether my method does better than feature counts
///use BIRL MAP solution as eval policy

using namespace std;

//Added stochastic transitions
////trying large scale experiment for feasible goal rewards
///trying with any random weights and no terminal state
///rewards that don't allow trajectories to the goal.
///using random world and random reward each time


//experiment7_1 has 200 reps, stochastic no duplicates steps 0.01 and 0.05 for 1:9 demos every 2, rollout 200, chain 10000

///TODO I realized that the feasibility is wrt to the number of demos, we should really
///try all possible demos so each run is equivalent, otherwise we'll have different rewards for different numbers of demos and some might be easier/harder and we wont get an apples to apples comparison...

enum World {SIMPLE, MAZE, CAKMAK1};

FeatureGridMDP* makeWorld(World w)
{
    FeatureGridMDP* fmdp = nullptr;
    
    if(w == SIMPLE)
    {   
    
        const int numFeatures = 2; //white, red, blue, green
        const int numStates = 9;
        const int width = 3;
        const int height = 3;
        double gamma = 0.95;
        vector<unsigned int> initStates = {0,1,2,3,4,5,6,7,8};
        vector<unsigned int> termStates = {6};
        bool stochastic = false;


        //go until can't improve (found local optima)


        //create random world //TODO delete it when done
        double** stateFeatures = improvementToyWorld(width, height, numFeatures, numStates);


        ///  create a random weight vector with seed and increment of rep number so same across reps
        double featureWeights[] = {-0.1, -0.9};      //white, red

        fmdp = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
        

        return fmdp;
    }
    else if(w == MAZE)
    {
        const int numFeatures = 2; //white, red, blue, green
        const int numStates = 25;
        const int width = 5;
        const int height = 5;
        double gamma = 0.95;
        vector<unsigned int> initStates;
        for(int i=0;i<numStates;i++)
            initStates.push_back(i);
        vector<unsigned int> termStates = {12};
        bool stochastic = false;


        //go until can't improve (found local optima)


        //create random world //TODO delete it when done
        double** stateFeatures = improvementMazeWorld(width, height, numFeatures, numStates);   


        ///  create a random weight vector with seed and increment of rep number so same across reps
        double featureWeights[] = {-0.1, -0.9};      //white, red

        fmdp = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
        

        return fmdp;
    
    
    }

    return fmdp;
}

int main() 
{
////////Most useful experiment params ////////////
//////////////    Change these ///////////////////

    double VaR = 0.99;
    World w = MAZE;
    vector<unsigned int> demoStates = {5};
/////////////////////////////

    string filename;
    if(w == SIMPLE)
        filename = "simple.txt";
    else if(w == MAZE)
        filename = "maze.txt";

    srand(time(NULL));

    //other Experiment parameters
  
    unsigned int rolloutLength = 100;          //max length of each demo
    double alpha = 100; //50                    //confidence param for BIRL
    const unsigned int chain_length = 10000;//1000;//5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    double step = 0.05; //0.01
    const double min_r = -1; //not used
    const double max_r = 1;  //not used
    bool removeDuplicates = true;
    double eps = 0.001;
    
    FeatureGridMDP* fmdp = makeWorld(w);
    
    vector<unsigned int> initStates;
    for(unsigned int s = 0; s < fmdp->getNumStates(); s++)
        if(fmdp->isInitialState(s))
            initStates.push_back(s);
    
    //TODO define this for each world
    vector<pair<unsigned int,unsigned int> > good_demos;
    vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
    

    //cout << "Transition function" << endl;
    //fmdp->displayTransitions();
    //cout << "-- Reward function --" << endl;
    //fmdp->displayRewards();
    ///  solve mdp for weights and get optimal policyLoss
    vector<unsigned int> opt_policy (fmdp->getNumStates());
    fmdp->valueIteration(eps);
    //cout << "-- value function ==" << endl;
    //fmdp->displayValues();
    cout << "features" << endl;
    displayStateColorFeatures(fmdp->getStateFeatures(), fmdp->getGridWidth(), fmdp->getGridHeight(), fmdp->getNumFeatures());
    //fmdp->deterministicPolicyIteration(opt_policy);
    fmdp->calculateQValues();
    fmdp->getOptimalPolicy(opt_policy);
    cout << "-- optimal policy --" << endl;
    fmdp->displayPolicy(opt_policy);

    cout << "-- feature weights --" << endl;
    fmdp->displayFeatureWeights();

    ///  generate demo
    trajectories.clear(); //used for feature counts
    for(unsigned int s0 : demoStates)
    {
       cout << "demo from " << s0 << endl;
       vector<pair<unsigned int, unsigned int>> traj = fmdp->monte_carlo_argmax_rollout(s0, rolloutLength);
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
                if(!fmdp->isTerminalState(p.first)) 
                    good_demos.push_back(p);
            }
    }
    
    //TODO write out the demo and the world in a format that I can use to show in plot


    ///  run BIRL to get chain and Map policyLoss ///
    //give it a copy of mdp to initialize
    FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
    birl.addPositiveDemos(good_demos);
    //birl.displayDemos();
    birl.run(eps);
    FeatureGridMDP* mapMDP = birl.getMAPmdp();
    mapMDP->displayFeatureWeights();
    cout << "Recovered reward" << endl;
    mapMDP->displayRewards();


    //make eval policy to be MAP policy
    vector<unsigned int> eval_pi (mapMDP->getNumStates()); 
    mapMDP->valueIteration(eps);  
    mapMDP->calculateQValues();         
    mapMDP->getOptimalPolicy(eval_pi);
    cout << "init policy" << endl;
    mapMDP->displayPolicy(eval_pi);
    //do inline and cheat...

    //Get V^* and V^\pi_eval for each start state 
    vector<vector<double>> evds(initStates.size(), vector<double>(chain_length));

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
        vector<double> Vstar_vec =  getExpectedReturnVector(sampleMDP);
        //cout << "True Exp Val" << endl;
        //cout << Vstar << endl;
        //cout << "Eval Policy" << endl; 
        vector<double> Vhat_vec = evaluateExpectedReturnVector(eval_pi, sampleMDP, eps);
        //cout << Vhat << endl;
        //save EVDiffs for each starting state for this hypothesis reward
        for(unsigned int j = 0; j < Vstar_vec.size(); j++)
        {
            double EVDiff = Vstar_vec[j] - Vhat_vec[j];
            evds[j][i] = EVDiff;
        }

    }    
    
    //output VaR data
    ofstream outfile_var("data/active/var_" + filename);
    for(unsigned int s = 0; s < evds.size(); s++)
    {
        std::sort(evds[s].begin(), evds[s].end());
        int VaR_index = (int) chain_length * VaR;
        double eval_VaR = evds[s][VaR_index];        
        if(s % fmdp->getGridWidth() < fmdp->getGridWidth() - 1)
            outfile_var << eval_VaR << ",";
        else
            outfile_var << eval_VaR << endl;
    }
    outfile_var.close();

    
    
    //TODO finish writing this out
    //output policy data
    ofstream outfile_policy("data/active/pi_" + filename);

    for(unsigned int s =0; s < evds.size(); s++)
    {
        if(fmdp->isTerminalState(s)) outfile_policy << ".";
        else if(fmdp->isWallState(s)) outfile_policy << "w";
        else if(eval_pi[s]==0) outfile_policy << "^";
        else if(eval_pi[s]==1) outfile_policy << "v";
        else if(eval_pi[s]==2) outfile_policy << "<";
        else if(eval_pi[s]==3) outfile_policy << ">";
        else cout << "ERROR, " << eval_pi[s] << " not a valid action!" << endl;
        if(s % fmdp->getGridWidth() < fmdp->getGridWidth() - 1)
            outfile_policy << ",";
        else
            outfile_policy << endl;
    }
    outfile_policy.close();


    
    cout << "------------"<< endl;
    //now for each start state EVD distribution, compute the one with the highest VaR
    for(unsigned int s = 0; s < evds.size(); s++)
    {
        cout << "computing VaR for initial state " << initStates[s] << endl;
        std::sort(evds[s].begin(), evds[s].end());
        int VaR_index = (int) chain_length * VaR;
        double eval_VaR = evds[s][VaR_index];
        cout << "VaR = " << eval_VaR << endl;
        
        
    

    }
    
    
    
    


    //clean up
    double** stateFeatures = fmdp->getStateFeatures();
     //delete features
    for(unsigned int s1 = 0; s1 < fmdp->getNumStates(); s1++)
    {
        delete[] stateFeatures[s1];
    }
    delete[] stateFeatures;
    
    delete fmdp;


}


