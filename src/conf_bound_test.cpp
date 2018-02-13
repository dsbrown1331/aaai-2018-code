#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/mdp.hpp"
#include "../include/confidence_bounds.hpp"

#define GRID_WIDTH 4
#define GRID_HEIGHT 4

#define MIN_R -2
#define MAX_R 2

#define CHAIN_LENGTH 6000

using namespace std;

int main( ) {

   vector<unsigned int> initStates;
   initStates.push_back(0);
   vector<unsigned int> termStates;
   termStates.push_back(15);
   
   GridMDP mdp(GRID_WIDTH,GRID_HEIGHT, initStates, termStates); //ground truth mdp
   //mdp.addTerminalState(terminal_state);
   cout << "\nInitializing gridworld of size " << GRID_WIDTH << " by " << GRID_HEIGHT << ".." << endl;
   cout << "    Num states: " << mdp.getNumStates() << endl;
   cout << "    Num actions: " << mdp.getNumActions() << endl;
//   cout << "    Terminals: "  << endl;
//   bool* terms = mdp.getTerminalStates();
//   for(int i=0;i<mdp.getNumStates();i++)
//       cout << i << ": " << terms[i] << endl;
   
   double* rewards = new double[mdp.getNumStates()]; //random reward function
   srand (time(NULL));
//   rewards[0] = 0;
//   rewards[1] = 1;
//   rewards[2] = 0;
//   rewards[3] = -1;
   for(unsigned int s=0; s<mdp.getNumStates(); s++)
   {
       rewards[s] = (rand()%20-20.0)/10.0; //forces all states except terminals to be negative
   }
   for(unsigned int t : termStates)
       rewards[t] = MAX_R;
   //cout << "Transitions" << endl;
   //mdp.displayTransitions();

   mdp.setRewards(rewards);
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
       GridMDP mdp_eval(GRID_WIDTH,GRID_HEIGHT, initStates, termStates);
       mdp_eval.setRewards(rewards);
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
      
   delete [] rewards;
    
   return 0;
}
