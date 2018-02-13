#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/mdp.hpp"
#include "../include/q_learner_grid.hpp"

#define GRID_WIDTH 5
#define GRID_HEIGHT 5

#define MIN_R -2
#define MAX_R 2

#define CHAIN_LENGTH 6000

using namespace std;

int main( ) {

   vector<unsigned int> initStates = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
   vector<unsigned int> termStates;
//   termStates.push_back(3);
   
   GridMDP mdp(GRID_WIDTH,GRID_HEIGHT, initStates, termStates, true); //ground truth mdp
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
   cout << "Transitions" << endl;
   mdp.displayTransitions();

   mdp.setRewards(rewards);
   cout << "\n-- True Rewards --" << endl;
   mdp.displayRewards();
   
   //solve for the optimal policy
   vector<unsigned int> opt_policy (mdp.getNumStates());
   mdp.valueIteration(0.001);
   cout << "-- value function ==" << endl;
   mdp.displayValues();
   //cout << "---- Q Values -----" << endl;
   //mdp.calculateQValues();
   //mdp.displayQValues();
   mdp.deterministicPolicyIteration(opt_policy);
   cout << "-- optimal policy --" << endl;
   mdp.displayPolicy(opt_policy);
   
   cout << "========== Q-learning =========" << endl;
   double learningRate = 0.01;
   double eps = 0.9;
   GridQLearner qlearn(&mdp);
   for(int i=0;i<5000;i++)
   {
       qlearn.trainEpoch(1000, eps, learningRate);
       //eps *= 0.999;
       //learningRate *= 0.99;
       //vector<unsigned int> q_policy (mdp.getNumStates());
       //qlearn.getArgMaxPolicy(q_policy);
       //qlearn.displayPolicy(q_policy);
       //cout << i << endl;
       //double base_loss = policyLoss(q_policy, &mdp);
       //cout << "Current policy loss: " << base_loss << "%" << endl;
   }


   //cout << "---- Q Values -----" << endl;
   //qlearn.displayQValues();
   cout << "-- q-argmax policy --" << endl;
   vector<unsigned int> q_policy (mdp.getNumStates());
   qlearn.getArgMaxPolicy(q_policy);
   qlearn.displayPolicy(q_policy);
   
   double base_loss = policyLoss(q_policy, &mdp);
   cout << "Current policy loss: " << base_loss << "%" << endl;
   
      
   delete [] rewards;
    
   return 0;
}
