#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/mdp.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/feature_birl.hpp"
#include "../include/grid_domains.hpp"



using namespace std;

int main( ) {

    const int numFeatures = 5; //white, red, blue, yellow, green
    const int numStates = 25;
    double gamma = 0.95;
    const int size = 5;
    double featureWeights[] = {0,-.5,+.5,0,0};
    
    double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {0,1,2,3,4,5,6,7,8,9,10,11,13,14,
                                        15,16,17,18,19,20,21,22,23,24};
    vector<unsigned int> termStates = {12};
   
    FeatureGridMDP mdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, gamma);

   cout << "\n-- True Rewards --" << endl;
   mdp.displayRewards();
   
   //solve for the optimal policy
   vector<unsigned int> opt_policy (mdp.getNumStates());
   mdp.valueIteration(0.001);
   cout << "-- value function ==" << endl;
   //mdp.displayValues();
   cout << "---- Q Values -----" << endl;
   mdp.calculateQValues();
   //mdp.displayQValues();
   mdp.deterministicPolicyIteration(opt_policy);
   cout << "-- optimal policy --" << endl;
   mdp.displayPolicy(opt_policy);
   
   for(int steps = 0; steps < 10; steps++)
   {
       //create copy of MDP
       FeatureGridMDP mdp_eval(mdp.getGridWidth(),mdp.getGridHeight(), mdp.getInitialStates(), mdp.getTerminalStates(), mdp.getNumFeatures(), mdp.getFeatureWeights(), mdp.getStateFeatures(), mdp.getDiscount());
       cout << "-- policy eval using " << steps << "--" << endl;
       //try a policy based on a certain number of policy iteration steps
       
       vector<unsigned int> evalPolicy (mdp_eval.getNumStates());
       mdp_eval.deterministicPolicyIteration(evalPolicy, steps);
       mdp_eval.displayPolicy(evalPolicy);
       double expV = evaluateExpectedReturn(evalPolicy, &mdp_eval, 0.001);
       cout << "eval policy expected return: " << expV << endl;
       double expVtrue = getExpectedReturn(&mdp); //uses existing Values
       cout << "optimal exp return: " << expVtrue << endl;
   }    
      
   delete stateFeatures;
    
   return 0;
}
