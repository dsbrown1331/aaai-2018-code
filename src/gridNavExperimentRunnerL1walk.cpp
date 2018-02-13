#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/mdp.hpp"
#include "../include/feature_birl.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include <fstream>
#include <string>


using namespace std;

int main( ) {
    
    const unsigned int grid_width = 5;
    const unsigned int grid_height = 5;
    const double min_r = -1.0;
    const double max_r = 1.0;
    const double step = 0.1;
    const double alpha = 1;
    const unsigned int chain_length = 5000;
    const unsigned int reps = 100;
    const int sample_flag = 2;   
    const bool mcmc_reject_flag = true;
    vector<int> step_list = {0,1,2,3,4};
    vector<vector<int>> demo_states = {{0},{1,4},{0,4,20,24},
                                {0,2,4,10,14,20,22,24},
                                {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                16,17,18,19,20,21,22,23,24}};
    
    
    srand (time(NULL));

    
    //test arrays to get features
    const int numFeatures = 5; //white, red, blue, yellow, green
    const int numStates = grid_width * grid_height;
    double gamma = 0.95;
    double featureWeights[] = {0,-0.5,+0.5,0,0};
    double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = randomGridNavDomain(numStates, numFeatures);
    cout << "---- State Features ----" << endl;
    displayStateColorFeatures(stateFeatures, grid_width, grid_height, numFeatures);
    
    
    //set up terminals and inits
    vector<unsigned int> initStates (numStates);
    for(unsigned int i=0; i<numStates; i++) initStates.push_back(i);
    vector<unsigned int> termStates = {12};


////////////////////
//Ground truth mdp
///////////////////  
    FeatureGridMDP mdp(grid_width, grid_height, initStates, termStates, numFeatures, featureWeights, stateFeatures, gamma);

cout << "\nInitializing gridworld of size " << grid_width << " by " << grid_height << ".." << endl;
   cout << "    Num states: " << mdp.getNumStates() << endl;
   cout << "    Num actions: " << mdp.getNumActions() << endl;
//   cout << "    Terminals: "  << endl;
//   bool* terms = mdp.getTerminalStates();
//   for(int i=0;i<mdp.getNumStates();i++)
//       cout << i << ": " << terms[i] << endl;
   

   
   cout << "\n-- True Rewards --" << endl;
   mdp.displayRewards();
   
   //solve for the optimal policy
   vector<unsigned int> opt_policy (mdp.getNumStates());
   mdp.valueIteration(0.001);
   cout << "-- value function ==" << endl;
   mdp.displayValues();
   mdp.deterministicPolicyIteration(opt_policy);
   cout << "-- optimal policy --" << endl;
   mdp.displayPolicy(opt_policy);
       


//iterate over different eval policies based on policy iteration

for(unsigned int demo_cnt = 0; demo_cnt < demo_states.size(); demo_cnt++)
{
   ///////////////////////////// 
   //generate demos
   /////////////////////////////
   vector<pair<unsigned int,unsigned int> > good_demos;
   mdp.calculateQValues();
   for(int d : demo_states[demo_cnt])
   {
       vector<pair<unsigned int, unsigned int>> demo = mdp.monte_carlo_argmax_rollout(d, 6);
       //for(pair<unsigned int, unsigned int> p : demo)
       //    cout << "(" <<  p.first << "," << p.second << ")" << endl;
       for(pair<unsigned int, unsigned int> p : demo)
           good_demos.push_back(p);
   }
   int states_demoed = demo_states[demo_cnt].size();

   cout << "----------------------------------------------------" << endl;
   cout << "num states in demo " << states_demoed << endl;
   cout << "----------------------------------------------------" << endl;

    for(int steps : step_list)
    {
        ////////////////////////////
        ////set up the evaluation policy
        //////////////////////////
        //cout << "using CTR policy" << endl;
        //vector<unsigned int> eval_pi(mdp.getNumStates());
        //for(unsigned int i=0; i<mdp.getNumStates();i++)
        //    eval_pi[i] = 3;
        
        cout << "using " << steps <<  " step policy eval" << endl;
        vector<unsigned int> eval_pi(mdp.getNumStates());
        //create copy of mdp and run n-step policy eval on it
        FeatureGridMDP mdp_eval(mdp.getGridWidth(),mdp.getGridHeight(), mdp.getInitialStates(), mdp.getTerminalStates(), mdp.getNumFeatures(), mdp.getFeatureWeights(), mdp.getStateFeatures(), mdp.getDiscount());
        mdp_eval.deterministicPolicyIteration(eval_pi, steps);
        //mdp_eval.displayPolicy(eval_pi);

        
        for(unsigned int rep = 0; rep < reps; rep++)
        {
            
            //set up file for output
            string filename = "gridNavToy_PIsteps" + to_string(steps) + "_numdemos" +  to_string(states_demoed)  + "_alpha" + to_string((int)alpha) + 
            "_chain" + to_string(chain_length) + "_L1sampleflag" + 
            to_string(sample_flag) + "_rep" + to_string(rep)+ ".txt";
            cout << filename << endl; 
            ofstream outfile("data/" + filename);
        

           

           
           
           //////////////////
           //create feature birl and initialize with demos
           ////////////////////
           //give it a copy of mdp to initialize
           FeatureGridMDP mdp_init(mdp.getGridWidth(),mdp.getGridHeight(), mdp.getInitialStates(), mdp.getTerminalStates(), mdp.getNumFeatures(), mdp.getFeatureWeights(), mdp.getStateFeatures(), mdp.getDiscount());
           FeatureBIRL birl(&mdp_init, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag);
           birl.addPositiveDemos(good_demos);
           //birl.displayDemos();
           //run birl MCMC
           //clock_t c_start = clock();
           birl.run();
           //clock_t c_end = clock();
           //cout << "\n[Timing] Time passed: "
           //           << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
           //FeatureGridMDP* mapMDP = birl.getMAPmdp();
           //mapMDP->displayFeatureWeights();
           //cout << "Recovered reward" << endl;
           //mapMDP->displayRewards();
           
           //solve for the optimal policy
           //vector<unsigned int> map_policy (mapMDP->getNumStates());
           //mapMDP->valueIteration(0.001);
           //cout << "-- value function ==" << endl;
           //mapMDP->displayValues();
           //mapMDP->deterministicPolicyIteration(map_policy);
           //cout << "-- optimal policy --" << endl;
           //mapMDP->displayPolicy(map_policy);
           //cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
           //double base_loss = policyLoss(map_policy, &mdp);
           //cout << "Current policy loss: " << base_loss << "%" << endl;
              
            int chainLen = birl.getChainLength();
            //cout<< "number of rewards in chain " << chainLen << endl;
            
            //Calculate differences and output them to file in format true\n---\ndata
            double trueDiff = abs(getExpectedReturn(&mdp) - evaluateExpectedReturn(eval_pi, &mdp, 0.001));
            outfile << "#true value --- mcmc ratios" << endl;
            outfile << trueDiff << endl;
            outfile << "---" << endl;
            
            for(int i=0; i<chainLen; i++)
            {
                //cout.precision(5);
                //get sampleMDP from chain
                GridMDP* sampleMDP = (*(birl.getRewardChain() + i));
                //((FeatureGridMDP*)sampleMDP)->displayFeatureWeights();
                //cout << "===================" << endl;
                //cout << "Reward " << i << endl;
                //sampleMDP->displayRewards();
                //cout << "--------" << endl;
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
                double Vhat = evaluateExpectedReturn(eval_pi, sampleMDP, 0.001);
                //cout << Vhat << endl;
                double VabsDiff = abs(Vstar - Vhat);
                //cout << "abs diff: " << VabsDiff << endl;
                outfile << VabsDiff << endl;
            }    
           
        }   
    } 
 }  
   //clean up memory for worlds states
   for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        delete[] stateFeatures[s1];
    }
    delete[] stateFeatures;
    
   return 0;
}
