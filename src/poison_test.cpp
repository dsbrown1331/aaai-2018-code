#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/feature_birl.hpp"

using namespace std;



int main() 
{
    srand(time(NULL));
    //test arrays to get features
    const int numFeatures = 3; //white, green blue
    const int numStates = 4;
    const int size = 2;
    double featureWeights[] = {0, -0.5, 0.5};
    //double featureWeights[] = {0,-0.5,+0.5};
    double gamma = 0.95;

    
    const double min_r = -1.0;
    const double max_r = 1.0;
    const double step = 0.1;
    const double alpha = 1;
    const unsigned int chain_length = 20000;
    const int sample_flag = 4;   
    const int num_steps = 10;
    const bool mcmc_reject_flag = true;

    double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {0,1,2};
    vector<unsigned int> termStates = {3};
    bool stochastic = false;
   
    FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);

    cout << "\nInitializing feature gridworld of size " << size << " by " << size << ".." << endl;
    cout << "    Num states: " << fmdp.getNumStates() << endl;
    cout << "    Num actions: " << fmdp.getNumActions() << endl;

    cout << " Features" << endl;

    //displayStateColorFeatures(stateFeatures, 5, 5, numFeatures);

    cout << "\n-- True Rewards --" << endl;
    fmdp.displayRewards();

    //solve for the optimal policy
    vector<unsigned int> opt_policy (fmdp.getNumStates());
    fmdp.valueIteration(0.001);
    cout << "-- value function ==" << endl;
    fmdp.displayValues();
    fmdp.getOptimalPolicy(opt_policy);
    cout << "-- optimal policy --" << endl;
    fmdp.displayPolicy(opt_policy);
    fmdp.calculateQValues();
    
   //generate adversarial demos
   vector<pair<unsigned int,unsigned int> > good_demos;
 
   good_demos.push_back(make_pair(2,3));
   good_demos.push_back(make_pair(0,3));
   good_demos.push_back(make_pair(1,1));

// crazy trajectory to put BIRL off its scent
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,3));
//   good_demos.push_back(make_pair(1,2));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,3));      
//   good_demos.push_back(make_pair(1,2));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,3));
//   good_demos.push_back(make_pair(1,2));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,3));
//   good_demos.push_back(make_pair(1,2));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,3));
//   good_demos.push_back(make_pair(1,2));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,0));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,2));
//   good_demos.push_back(make_pair(0,3));
//   good_demos.push_back(make_pair(1,0));
//   good_demos.push_back(make_pair(1,3));
//   good_demos.push_back(make_pair(1,0));
//   good_demos.push_back(make_pair(1,3));
//   good_demos.push_back(make_pair(1,1));   
//   good_demos.push_back(make_pair(3,2));
   
   
   
   //create feature birl and initialize with demos
   FeatureBIRL birl(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
   birl.addPositiveDemos(good_demos);
   birl.displayDemos();
   //run birl MCMC
   birl.run();
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
   mapMDP->calculateQValues();
   mapMDP->getOptimalPolicy(map_policy);
   cout << "-- optimal policy --" << endl;
   mapMDP->displayPolicy(map_policy);
   
   cout <<"Qvals" << endl;
   mapMDP->displayQValues();
    cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
    cout << "Freeing variables" << endl;
    for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        delete[] stateFeatures[s1];
        //delete[] stateFeatureCnts[s1];
    }
    delete[] stateFeatures;
    //delete[] stateFeatureCnts;
    //delete[] expFeatureCnts;


}


