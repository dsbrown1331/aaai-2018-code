#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/mdp.hpp"
#include "../include/birl.hpp"

#define GRID_WIDTH 5
#define GRID_HEIGHT 5

#define MIN_R -2
#define MAX_R 2

#define CHAIN_LENGTH 5000


using namespace std;

const double step = 0.1; //step size for grid walk in MCMC
const double alpha = 10; //higher values put more confidence on demonstrations


int main( ) {

   unsigned int terminal_state = GRID_WIDTH - 1;
   //set up initial and terminal states, initial doesn't affect algorithm but could be used to determine where to give demos from in the future
   vector<unsigned int> initStates = {};
   vector<unsigned int> termStates = {terminal_state};
   
   GridMDP mdp(GRID_WIDTH, GRID_HEIGHT, initStates, termStates); //ground truth mdp
   
   cout << "\nInitializing gridworld of size " << GRID_WIDTH << " by " << GRID_HEIGHT << ".." << endl;
   cout << "    Num states: " << mdp.getNumStates() << endl;
   cout << "    Num actions: " << mdp.getNumActions() << endl;
   
   double* rewards = new double[mdp.getNumStates()]; //random reward function
   srand (time(NULL));
   for(unsigned int s=0; s<mdp.getNumStates(); s++)
   {
       rewards[s] = (rand()%20-20.0)/10.0; //forces all states except terminals to be negative
   }
   rewards[terminal_state] = MAX_R;
   mdp.setRewards(rewards);
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
   
   //initializing birl with reward ranges, BIRL chain length and reward step size
   
   vector<unsigned int> policy (mdp.getNumStates());
   BIRL birl(GRID_WIDTH,GRID_HEIGHT, initStates, termStates, MIN_R, MAX_R, CHAIN_LENGTH, step, alpha);
   
   vector<pair<unsigned int,unsigned int> > good_demos;
   vector<pair<unsigned int,unsigned int> > bad_demos;
   mdp.calculateQValues();
   for(unsigned int s=0; s < mdp.getNumStates(); s++)
   {
      //int rand_state = rand() % mdp.getNumStates();
      //good_demos.push_back(make_pair(idx, opt_policy[idx])); //ground truth policy as input
      for (unsigned int a=0; a < mdp.getNumActions(); a++)
      {
          //if (a != opt_policy[idx]) bad_demos.push_back(make_pair(idx, a));
          if(mdp.isOptimalAction(s,a) ) good_demos.push_back(make_pair(s,a));
          else bad_demos.push_back(make_pair(s, a));
      }
   }
   
   birl.addPositiveDemos(good_demos);
   birl.addNegativeDemos(bad_demos);
   birl.displayDemos();
   
   clock_t c_start = clock();
   birl.run();
   clock_t c_end = clock();
   cout << "\n[Timing] Time passed: "
              << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
              
   /*if( mdp == birl.getMDP() ) cout << "Two rewards are equal!" << endl;
   else cout << "Two rewards are not equal!" << endl;*/
    
    cout << "\n-- Final Recovered Rewards --" << endl;
    birl.getMAPmdp()->displayRewards();
    cout << "\n-- Final Policy --" << endl;
    birl.getMAPmdp()->deterministicPolicyIteration(policy);
    birl.getMAPmdp()->displayPolicy(policy);
    
    cout.precision(12);
    cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
    cout.precision(2);
    double base_loss = policyLoss(policy, &mdp);
    cout << "Current policy loss: " << base_loss << "%" << endl;
    
    int chainLen = birl.getChainLength();
    cout<< "number of rewards in chain " << chainLen << endl;
    
//    for(int i=0; i<chainLen; i++)
//    {
//        cout << "===================" << endl;
//        cout << "Reward " << i << endl;
//        (*(birl.getRewardChain() + i))->displayRewards();
//        cout << "--------" << endl;
//        cout << "Value" << endl;
//        (*(birl.getRewardChain() + i))->displayValues();
//        cout << "-----------" << endl;
//        (*(birl.getRewardChain() + i))->displayQValues();        

//        
//    }
    
    delete [] rewards;
    
    return 0;
}
