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


using namespace std;

int main( ) {
    
    const unsigned int grid_width = 5;
    const unsigned int grid_height = 5;
    const double min_r = -1.0;
    const double max_r = 1.0;
    const double step = 0.1;
    const double alpha = 50;
    const unsigned int chain_length = 10000;
    
    //test arrays to get features
    const int numFeatures = 5; //white, red, blue, yellow, green
    const int numStates = grid_width * grid_height;
    double gamma = 0.95;
    double featureWeights[] = {0,-1,+1,0,0};
    double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = randomGridNavDomain(numStates, numFeatures);
    cout << "---- State Features ----" << endl;
    displayStateColorFeatures(stateFeatures, grid_width, grid_height, numFeatures);
    
    //set up file for output
    ofstream outfile("test.txt");
    
    
    //set up terminals and inits
    vector<unsigned int> initStates (numStates);
    for(unsigned int i=0; i<numStates; i++) initStates.push_back(i);
    vector<unsigned int> termStates = {12};

  
    FeatureGridMDP mdp(grid_width, grid_height, initStates, termStates, numFeatures, featureWeights, stateFeatures, gamma);

    //set up the evaluation policy
    cout << "using CTR policy" << endl;
    vector<unsigned int> eval_pi(mdp.getNumStates());
    for(unsigned int i=0; i<mdp.getNumStates();i++)
        eval_pi[i] = 3;



   cout << "\nInitializing gridworld of size " << grid_width << " by " << grid_height << ".." << endl;
   cout << "    Num states: " << mdp.getNumStates() << endl;
   cout << "    Num actions: " << mdp.getNumActions() << endl;
//   cout << "    Terminals: "  << endl;
//   bool* terms = mdp.getTerminalStates();
//   for(int i=0;i<mdp.getNumStates();i++)
//       cout << i << ": " << terms[i] << endl;
   
   srand (time(NULL));
   
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
   
   cout << "implement a policy iteration algorithm that runs for only x steps to get incrementally better policies" << endl;
   
   //generate demos
   vector<pair<unsigned int,unsigned int> > good_demos;
   mdp.calculateQValues();
   for(unsigned int s=0; s < mdp.getNumStates(); s++)
   {
      //int rand_state = rand() % mdp.getNumStates();
      //good_demos.push_back(make_pair(idx, opt_policy[idx])); //ground truth policy as input
      for (unsigned int a=0; a < mdp.getNumActions(); a++)
      {
          //if (a != opt_policy[idx]) bad_demos.push_back(make_pair(idx, a));
          if(mdp.isOptimalAction(s,a) ) 
            good_demos.push_back(make_pair(s,a));

      }
   }
   
   

   
   
   //create feature birl and initialize with demos
   FeatureBIRL birl(&mdp, min_r, max_r, chain_length, step, alpha);
   birl.addPositiveDemos(good_demos);
   birl.displayDemos();
   //run birl MCMC
   clock_t c_start = clock();
   birl.run();
   clock_t c_end = clock();
   cout << "\n[Timing] Time passed: "
              << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
   FeatureGridMDP* mapMDP = birl.getMAPmdp();
   
   cout << "Recovered reward" << endl;
   mapMDP->displayRewards();
   
   //solve for the optimal policy
   vector<unsigned int> map_policy (mapMDP->getNumStates());
   mapMDP->valueIteration(0.001);
   cout << "-- value function ==" << endl;
   mapMDP->displayValues();
   mapMDP->deterministicPolicyIteration(map_policy);
   cout << "-- optimal policy --" << endl;
   mapMDP->displayPolicy(map_policy);
   
   
   cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
   
   double base_loss = policyLoss(map_policy, &mdp);
   cout << "Current policy loss: " << base_loss << "%" << endl;
      
    int chainLen = birl.getChainLength();
    cout<< "number of rewards in chain " << chainLen << endl;
    
    //Calculate differences and output them to file in format true\n---\ndata
    double trueDiff = getExpectedReturn(&mdp) - evaluateExpectedReturn(eval_pi, &mdp, 0.001);
    outfile << "#true value --- mcmc ratios" << endl;
    outfile << trueDiff << endl;
    outfile << "---" << endl;
    
    for(int i=0; i<chainLen; i++)
    {
        //cout.precision(5);
        //get sampleMDP from chain
        GridMDP* sampleMDP = (*(birl.getRewardChain() + i));
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
    
   
   //clean up memory
   for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        delete[] stateFeatures[s1];
    }
    delete[] stateFeatures;
    
   return 0;
}
