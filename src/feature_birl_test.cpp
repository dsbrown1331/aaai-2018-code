#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/mdp.hpp"
#include "../include/feature_birl.hpp"
#include "../include/grid_domains.hpp"


using namespace std;

int main( ) {
    const unsigned int grid_width = 4;
    const unsigned int grid_height = 4;
    const double min_r = -1.0;
    const double max_r = 1.0;
    const double step = 0.02;
    const double alpha = 10;
    const unsigned int chain_length = 1000;
    const int sample_flag = 4;   
    const int num_steps = 10;
    const bool mcmc_reject_flag = true;
    int skip = 100;
    int burn = 100;
    bool stochastic = true;
    
    //0 is normal grid walk, 1 is random l1-unit-ball, 2 is randomly tweaking two dims
    
    //test arrays to get features
    const int numFeatures = 4; //white, red, blue, yellow, green
    const int numStates = grid_width * grid_height;
    //double gamma = 0.95;
    //double featureWeights[] = {0,-0.5,+0.5,0,0};
    double gamma = 0.99;
    double featureWeights[] = {-0.1,-0.1,-0.4,+0,4};
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    double** stateFeatures = initFeatureCountToyDomain4x4(numStates, numFeatures);
    
    //set up terminals and inits
    ////vector<unsigned int> initStates = {};
    ////vector<unsigned int> termStates = {12};
   vector<unsigned int> initStates = {0};
   vector<unsigned int> termStates = {15};
    FeatureGridMDP mdp(grid_width, grid_height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);


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

//   good_demos.push_back(make_pair(0,1));
//   good_demos.push_back(make_pair(4,1));
//   good_demos.push_back(make_pair(8,1));
//   good_demos.push_back(make_pair(12,3));
//   good_demos.push_back(make_pair(13,3));   
//   good_demos.push_back(make_pair(14,3));   
//   good_demos.push_back(make_pair(15,3));
//   
//   good_demos.push_back(make_pair(0,3));
//   good_demos.push_back(make_pair(1,3));
//   good_demos.push_back(make_pair(2,3));
//   good_demos.push_back(make_pair(3,1));
//   good_demos.push_back(make_pair(7,1));   
//   good_demos.push_back(make_pair(11,1));   
//   good_demos.push_back(make_pair(15,3));
   
   
   //create feature birl and initialize with demos
   FeatureBIRL birl(&mdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
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
   cout << "Feature weights" << endl;
   mapMDP->displayFeatureWeights();
   
   //solve for the optimal policy
   vector<unsigned int> map_policy (mapMDP->getNumStates());
   mapMDP->valueIteration(0.001);
   cout << "-- value function ==" << endl;
   mapMDP->displayValues();
   mapMDP->deterministicPolicyIteration(map_policy);
   cout << "-- optimal policy --" << endl;
   mapMDP->displayPolicy(map_policy);
   
   //cout.precision(12);
   cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
   //cout.precision(2);
   double base_loss = policyLoss(map_policy, &mdp);
   cout << "Current policy loss: " << base_loss << "%" << endl;
      
    int chainLen = birl.getChainLength();
    cout<< "number of rewards in chain " << chainLen << endl;
    
    for(int i=burn; i<chainLen; i+=skip)
    {
        cout << "===================" << endl;
        cout << "Reward " << i << endl;
//        (*(birl.getRewardChain() + i))->displayRewards();
        cout << "weights" << endl;
        (*(birl.getRewardChain() + i))->displayFeatureWeights();
//        vector<unsigned int> sample_policy (mapMDP->getNumStates());
//        (*(birl.getRewardChain() + i))->getOptimalPolicy(sample_policy);
//        cout << "optimal policy" << endl;
//        (*(birl.getRewardChain() + i))->displayPolicy(sample_policy);
//        cout << "--------" << endl;
//        cout << "Value" << endl;
//        (*(birl.getRewardChain() + i))->displayValues();
//        cout << "-----------" << endl;
//        (*(birl.getRewardChain() + i))->displayQValues();        

        
    }  

   FeatureGridMDP* mean_mdp = birl.getMeanMDP(burn, skip);
   cout << "MEAN reward" << endl;
   mean_mdp->displayRewards();
   cout << "Feature weights" << endl;
   mean_mdp->displayFeatureWeights();
   
   //solve for the optimal policy
   vector<unsigned int> mean_policy (mean_mdp->getNumStates());
   mean_mdp->valueIteration(0.001);
   cout << "-- value function ==" << endl;
   mean_mdp->displayValues();
   mean_mdp->deterministicPolicyIteration(mean_policy);
   cout << "-- optimal policy --" << endl;
   mean_mdp->displayPolicy(mean_policy);


   
   
   //clean up memory
   for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        delete[] stateFeatures[s1];
    }
    delete[] stateFeatures;
    
   return 0;
}
