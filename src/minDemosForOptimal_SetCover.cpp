#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include "../include/optimalTeaching.hpp"
#include <fstream>
#include <string>

///test out whether my method does better than feature counts
///use BIRL MAP solution as eval policy

using namespace std;


///trying with non-negative weight on goal and non-positive weight everywhere else but weeding out
///rewards that don't allow trajectories to the goal.
///experiment 4_1 on my laptop is to try out a fixed world with random feasible rewards for go to goal behavior


int main() 
{

    ////Experiment parameters
    const unsigned int reps = 10;                    //repetitions per setting
    unsigned int numDemo = 4;            //number of demos to give
    unsigned int rolloutLength = 30;          //max length of each demo
    int K = 1; //number of demos from each state
    double alpha = 100; //50                    //confidence param for BIRL
    const unsigned int chain_length = 5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    double step = 0.01; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool removeDuplicates = true;
    bool stochastic = false;
    
    int startSeed = 132;
    cout << "start seed = " << startSeed << endl;
    double eps = 0.0001;
    
    //test arrays to get features
    const int numFeatures = 4; //white, red, blue, green
    //const int numStates = 81;
    const int size = 7;
    double gamma = 0.95;
    double** stateFeatures = random7x7GridNavGoalWorld4Features();
    //vector<unsigned int> initStates = {8,10,12,22,26,36,38,40};
    vector<unsigned int> initStates = {8,12,36,40};
    vector<unsigned int> termStates = {24};

    //set up file for output
    string filename = "set_cover.txt";
    cout << filename << endl; 
    ofstream outfile("data/experiment_min_demos/" + filename);

    for(unsigned int rep = 0; rep < reps; rep++)
    {
        
    
        srand(startSeed + 3*rep);
        cout << "------Rep: " << rep << "------" << endl;



        bool isRewardFeasible = true;
        double featureWeights[numFeatures];
        vector<pair<unsigned int,unsigned int> > good_demos;
        vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
        int count = 0;
        ////Loop until feasible reward is found
        do{
            count++;
            cout << "count = " << count << endl;
            isRewardFeasible = true;
            ///  create a random weight vector with seed and increment of rep number so same across reps
            double* negFeatureWeights = sample_unit_L1_norm(numFeatures);
            
            //set weights for all features except goal feature to be non-positive
            for(int i=0;i<numFeatures;i++)
                featureWeights[i] = -abs(negFeatureWeights[i]);
            //set blue to be non-negative
            featureWeights[2] = abs(negFeatureWeights[2]);
            assert(isEqual(l1_norm(featureWeights, numFeatures),1.0));
            delete[] negFeatureWeights;
            FeatureGridMDP mdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);

            ///  solve mdp for weights and get optimal policyLoss
            vector<unsigned int> opt_policy (mdp.getNumStates());
            mdp.valueIteration(eps);
            //mdp.displayValues();
            cout << "-- rewards --" << endl;
            mdp.displayRewards();
            mdp.calculateQValues();
            mdp.getOptimalPolicy(opt_policy);
            cout << "-- optimal policy --" << endl;
            mdp.displayPolicy(opt_policy);
            cout << "-- feature weights --" << endl;
            mdp.displayFeatureWeights();

            ///  generate numDemo demos from the initial state distribution
            trajectories.clear(); //used for feature counts
            for(unsigned int d = 0; d < numDemo; d++)
            {
               unsigned int s0 = initStates[d];
               //cout << "demo from " << s0 << endl;
               vector<pair<unsigned int, unsigned int>> traj = mdp.policy_rollout(s0, rolloutLength, opt_policy);
               /////Check if reached the terminal state (blue)
               if(traj.size() == rolloutLength)
               {
                    isRewardFeasible = false;
                    break;
                }
               //cout << "trajectory " << d << endl;
               //for(pair<unsigned int, unsigned int> p : traj)
               //    cout << "(" <<  p.first << "," << p.second << ")" << endl;
               trajectories.push_back(traj);
            }


        }
        while(!isRewardFeasible);
        
        cout << "--Final reward chosen---" << endl;
        //create actual mdp for use using feasible reward
        FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
        fmdp.valueIteration(0.001);
        fmdp.calculateQValues();
        vector<unsigned int> opt_policy (fmdp.getNumStates());
        fmdp.getOptimalPolicy(opt_policy);
        cout << "-- rewards --" << endl;
        fmdp.displayRewards();
        cout << "-- optimal policy --" << endl;
        fmdp.displayPolicy(opt_policy);
        cout << "-- feature weights --" << endl;
        fmdp.displayFeatureWeights();
        cout << "trajectories" << endl;
        for(vector<pair<unsigned int, unsigned int>> traj : trajectories)
        {
            for(pair<unsigned int, unsigned int> p : traj)
               cout << "(" <<  p.first << "," << p.second << "), ";
            cout << endl;
        }    
        
        //Run set_cover optimal teaching and give to BIRL
        //get set cover solution
        cout << "====================" << endl;
        cout << "deterministic " << endl;
        vector<vector<pair<unsigned int, unsigned int> > > trajectories_sc = solveSetCoverOptimalTeaching(&fmdp, K, rolloutLength, false);       
         cout << "----- Solution ------" << endl;
        int d_count = 0;
        //print out demos
        for(vector<pair<unsigned int, unsigned int> > traj : trajectories_sc)
        {
            cout << "demo " << d_count << endl;
            d_count++;
            for(pair<unsigned int, unsigned int> p : traj)
                cout << "(" <<  p.first << "," << p.second << "), ";
            cout << endl;
        }
        
        //remove duplicates and put in good_demos
        for(vector<pair<unsigned int, unsigned int> > traj : trajectories_sc)
        {
            for(pair<unsigned int, unsigned int> p : traj)
            {
                if(removeDuplicates)
                {
                    if(std::find(good_demos.begin(), good_demos.end(), p) == good_demos.end())
                        good_demos.push_back(p);
                }
                else
                {    
                    good_demos.push_back(p);
                }
            }
        }
    
       
        
        ///  run BIRL to get chain and Map policyLoss ///
        //give it a copy of mdp to initialize
        FeatureBIRL birl(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
        birl.addPositiveDemos(good_demos);
        birl.displayDemos();
        birl.run();
        FeatureGridMDP* mapMDP = birl.getMAPmdp();
        mapMDP->displayFeatureWeights();
        //cout << "Recovered reward" << endl;
        //mapMDP->displayRewards();

        //solve for the optimal policy
        vector<unsigned int> eval_pi (mapMDP->getNumStates());
        mapMDP->valueIteration(eps);
        mapMDP->calculateQValues();
        mapMDP->getOptimalPolicy(eval_pi);
//                cout << "-- value function ==" << endl;
//                mapMDP->displayValues();
//                mapMDP->deterministicPolicyIteration(map_policy);
        cout << "-- learned policy --" << endl;
        mapMDP->displayPolicy(eval_pi);
        //cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
        //double base_loss = policyLoss(eval_pi, &fmdp);
        //cout << "Current policy loss: " << base_loss << "%" << endl;

        /// We use the Map Policy as the evaluation policy

        
        
        //write actual, worst-case, and chain info to file

        ///compute actual expected return difference
        double trueDiff = abs(getExpectedReturn(&fmdp) - evaluateExpectedReturn(eval_pi, &fmdp, eps));
        cout << "True difference: " << trueDiff << endl;
        //double actionLoss = policyLoss(eval_pi, &fmdp);
        //cout << "0-1 Loss: " << actionLoss << endl;
        outfile << trajectories_sc.size() << ",";
        outfile << trueDiff << "\n";
        
        
        
         
             
    }
}


