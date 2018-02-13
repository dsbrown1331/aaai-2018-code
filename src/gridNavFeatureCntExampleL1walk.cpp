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


//test out ideas for feature counts versus my method

using namespace std;

int main( ) {
    
    const unsigned int grid_width = 4;
    const unsigned int grid_height = 4;
    const double min_r = -1.0;
    const double max_r = 1.0;
    const double step = 0.02;
    const double alpha = 1;
    const unsigned int chain_length = 10000;
    const unsigned int reps = 100;
    const int sample_flag = 4;   
    const int num_steps = 10;
    const bool mcmc_reject_flag = true;
    
    
    srand (time(NULL));

    
    //test arrays to get features
    const int numFeatures = 4; //white, red, blue, yellow, green
    const int numStates = grid_width * grid_height;
    double gamma = 0.99;
    double featureWeights[] = {-0.1,-0.1,-0.4,+0,4};
//    double featureWeights[] = {-0.18, -0.18, -0.64, 0.00}; //MLE reward for true policy
//    double featureWeights[] = {-0.16, -0.16, -0.55, -0.03}; //Mean reward
    //double featureWeights[] = {0, 0, -0.5, 0.5}; //other reward
    double** stateFeatures = initFeatureCountToyDomain4x4(numStates, numFeatures);
    //double** stateFeatures = randomGridNavDomain(numStates, numFeatures);
    cout << "---- State Features ----" << endl;
    displayStateColorFeatures(stateFeatures, grid_width, grid_height, numFeatures);
    
    
    //set up terminals and inits
    vector<unsigned int> initStates ={0};
    vector<unsigned int> termStates = {15};


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

///////////////////////////// 
//generate demos
/////////////////////////////
vector<pair<unsigned int,unsigned int> > good_demos;
mdp.calculateQValues();

   good_demos.push_back(make_pair(0,1));
   good_demos.push_back(make_pair(4,1));
   good_demos.push_back(make_pair(8,1));
   good_demos.push_back(make_pair(12,3));
   good_demos.push_back(make_pair(13,3));   
   good_demos.push_back(make_pair(14,3));   
   good_demos.push_back(make_pair(15,3));
   
   good_demos.push_back(make_pair(0,3));
   good_demos.push_back(make_pair(1,3));
   good_demos.push_back(make_pair(2,3));
   good_demos.push_back(make_pair(3,1));
   good_demos.push_back(make_pair(7,1));   
   good_demos.push_back(make_pair(11,1));   
   good_demos.push_back(make_pair(15,3));
   

////////////////////////////
////set up the evaluation policy
//////////////////////////
cout << "Evaluation Policy" << endl;
vector<unsigned int> eval_pi(mdp.getNumStates());
////bad eval policy
unsigned int actions[4][4] = {{1,1,1,1},
                            {1,1,1,1},
                            {3,1,1,1},
                            {0,3,3,3}};
 
 
////good eval policy                           
//unsigned int actions[4][4] = {{3,3,3,1},
//                            {0,0,0,1},
//                            {0,0,0,1},
//                            {0,0,0,1}};
int cnt = 0;
for(int i=0; i<4;i++)
{
    for(int j=0; j<4;j++)
    {
        //cout << actions[i][j] << endl;
        eval_pi[cnt] = actions[i][j];
        cnt += 1;
    }    
}
mdp.displayPolicy(eval_pi);


for(unsigned int rep = 0; rep < reps; rep++)
{
    
    //set up file for output
    string filename = "fcount_badeval_alpha" + to_string((int)alpha) + 
    "_chain" + to_string(chain_length) + "_L1sampleflag" + 
    to_string(sample_flag) + "_steps" + to_string(num_steps) + "_rep" + to_string(rep)+ ".txt";
    cout << filename << endl; 
    ofstream outfile("data/fcountAlpha1/" + filename);


   

   
   
   //////////////////
   //create feature birl and initialize with demos
   ////////////////////
   //give it a copy of mdp to initialize
   FeatureGridMDP mdp_init(mdp.getGridWidth(),mdp.getGridHeight(), mdp.getInitialStates(), mdp.getTerminalStates(), mdp.getNumFeatures(), mdp.getFeatureWeights(), mdp.getStateFeatures(), mdp.getDiscount());
   FeatureBIRL birl(&mdp_init, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
   birl.addPositiveDemos(good_demos);
   birl.displayDemos();
   //run birl MCMC
   clock_t c_start = clock();
   birl.run();
   clock_t c_end = clock();
   cout << "\n[Timing] Time passed: "
              << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n";
   FeatureGridMDP* mapMDP = birl.getMAPmdp();
   mapMDP->displayFeatureWeights();
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
    //cout<< "number of rewards in chain " << chainLen << endl;
    
    //Calculate differences and output them to file in format true\n---\ndata
    double trueDiff = abs(getExpectedReturn(&mdp) - evaluateExpectedReturn(eval_pi, &mdp, 0.001));
    outfile << "#true value --- mcmc ratios" << endl;
    outfile << trueDiff << endl;
    outfile << "---" << endl;
    
    //debugging
    cout << "#true value --- mcmc ratios" << endl;
    cout << trueDiff << endl;
    cout << "---" << endl;
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
        //TODO: rollouts might be more efficient than fully solving mdp
        double Vhat = evaluateExpectedReturn(eval_pi, sampleMDP, 0.001);
        //cout << Vhat << endl;
        double VabsDiff = abs(Vstar - Vhat);
        //cout << "abs diff: " << VabsDiff << endl;
        outfile << VabsDiff << endl;
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
