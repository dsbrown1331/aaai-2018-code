#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

///test out whether my method does better than feature counts
///use BIRL MAP solution as eval policy

using namespace std;

////trying large scale experiment for feasible goal rewards
///trying with any random weights and no terminal state
///rewards that don't allow trajectories to the goal.
///using random world and random reward each time
/// experiment6_1 on lab machine has no duplicates
/// experiment6_2 on lab machine has duplicates

/// experiment6_3 on lab machine has no duplicates and uses gamma = 0.9 with rolloutLength 100
/// experiment6_4 on lab machine has duplicates        '''

///TODO I realized that the feasibility is wrt to the number of demos, we should really
///try all possible demos so each run is equivalent, otherwise we'll have different rewards for different numbers of demos and some might be easier/harder and we wont get an apples to apples comparison...


int main() 
{

    ////Experiment parameters
    const unsigned int reps = 200;                    //repetitions per setting
    const vector<unsigned int> numDemos = {1,2,3,4,5,6,7,8,9};            //number of demos to give
    const vector<unsigned int> rolloutLengths = {100};          //max length of each demo
    const vector<double> alphas = {100, 10, 1}; //50                    //confidence param for BIRL
    const unsigned int chain_length = 10000;//1000;//5000;        //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    const vector<double> steps = {0.01}; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool removeDuplicates = false;

    int startSeed = 132;
    double eps = 0.001;
    
    //test arrays to get features
    const int numFeatures = 8; //white, red, blue, yellow, green
    const int numStates = 81;
    const int size = 9;
    double gamma = 0.9;
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);  
//    double** stateFeatures = random9x9GridNavGoalWorld();
//    double** stateFeatures = random9x9GridNavGoalWorld8Features();
    vector<unsigned int> initStates = {10, 13, 16, 37, 40, 43, 64, 67, 70};
    vector<unsigned int> termStates = {};

for(unsigned int rolloutLength : rolloutLengths)
{
    //iterate over alphas
    for(double alpha : alphas)
    {
        cout << "======Alpha: " << alpha << "=====" << endl;
        //iterate over number of demonstrations
        for(unsigned int numDemo : numDemos)
        {
            cout << "****Num Demos: " << numDemo << "****" << endl;
            //iterate over repetitions
        for(double step : steps)
        {
            for(unsigned int rep = 0; rep < reps; rep++)
            {
                //set up file for output
                string filename = "numdemos" +  to_string(numDemo) 
                                + "_alpha" + to_string((int)alpha) 
                                + "_chain" + to_string(chain_length) 
                                + "_step" + to_string(step)
                                + "_L1sampleflag" + to_string(sample_flag) 
                                + "_rolloutLength" + to_string(rolloutLength)
                                + "_rep" + to_string(rep)+ ".txt";
                cout << filename << endl; 
                ofstream outfile("data/experiment6_4/" + filename);
            
                srand(startSeed + 3*rep);
                cout << "------Rep: " << rep << "------" << endl;

                //create random world //TODO delete it when done
                double** stateFeatures = random9x9GridNavWorld8Features();
        
                vector<pair<unsigned int,unsigned int> > good_demos;
                vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
              
                ///  create a random weight vector with seed and increment of rep number so same across reps
                double* featureWeights = sample_unit_L1_norm(numFeatures);
                    
                FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, gamma);
                delete[] featureWeights;
                ///  solve mdp for weights and get optimal policyLoss
                vector<unsigned int> opt_policy (fmdp.getNumStates());
                fmdp.valueIteration(eps);
                //cout << "-- value function ==" << endl;
                //mdp.displayValues();
                cout << "features" << endl;
                displayStateColorFeatures(stateFeatures, size, size, numFeatures);
                fmdp.deterministicPolicyIteration(opt_policy);
                cout << "-- optimal policy --" << endl;
                fmdp.displayPolicy(opt_policy);
                fmdp.calculateQValues();
                cout << "-- feature weights --" << endl;
                fmdp.displayFeatureWeights();

                ///  generate numDemo demos from the initial state distribution
                trajectories.clear(); //used for feature counts
                for(unsigned int d = 0; d < numDemo; d++)
                {
                   unsigned int s0 = initStates[d];
                   //cout << "demo from " << s0 << endl;
                   vector<pair<unsigned int, unsigned int>> traj = fmdp.monte_carlo_argmax_rollout(s0, rolloutLength);
                   cout << "trajectory " << d << endl;
                   for(pair<unsigned int, unsigned int> p : traj)
                       cout << "(" <<  p.first << "," << p.second << ")" << endl;
                   trajectories.push_back(traj);
                }
                //put trajectories into one big vector for birl_test
                //weed out duplicate demonstrations
                good_demos.clear();
                for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
                    for(pair<unsigned int, unsigned int> p : traj)
                        if(removeDuplicates)
                        {
                            if(std::find(good_demos.begin(), good_demos.end(), p) == good_demos.end())
                                good_demos.push_back(p);
                        }
                        else
                        {    
                            good_demos.push_back(p);
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
//                cout << "-- optimal policy --" << endl;
                //mapMDP->displayPolicy(eval_pi);
                //cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
                //double base_loss = policyLoss(eval_pi, &fmdp);
                //cout << "Current policy loss: " << base_loss << "%" << endl;

                /// We use the Map Policy as the evaluation policy

                
                
                //write actual, worst-case, and chain info to file

                ///compute actual expected return difference
                double trueDiff = abs(getExpectedReturn(&fmdp) - evaluateExpectedReturn(eval_pi, &fmdp, eps));
                cout << "True difference: " << trueDiff << endl;
                outfile << "#true value --- wfcb --- mcmc ratios" << endl;
                outfile << trueDiff << endl;
                outfile << "---" << endl;
                //compute worst-case feature count bound
                double wfcb = calculateWorstCaseFeatureCountBound(eval_pi, &fmdp, trajectories, eps);
                cout << "WFCB: " << wfcb << endl;
                outfile << wfcb << endl;
                outfile << "---" << endl;

                
                 //Calculate differences and output them to file in format true\n---\ndata
                for(unsigned int i=0; i<chain_length; i++)
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
                    double Vhat = evaluateExpectedReturn(eval_pi, sampleMDP, eps);
                    //cout << Vhat << endl;
                    double VabsDiff = abs(Vstar - Vhat);
                    //cout << "abs diff: " << VabsDiff << endl;
                    outfile << VabsDiff << endl;
                }    
           

                //delete world
                for(unsigned int s1 = 0; s1 < numStates; s1++)
                {
                    delete[] stateFeatures[s1];
                }
                delete[] stateFeatures;

            }

        }

        }
    }
}
//    double featureWeights[] = {0,-1,+1,0,0};
//    


//    //set up terminals and inits

//    vector<unsigned int> demoStates = initStates;
//   
//    FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, gamma);

//    cout << "\nInitializing feature gridworld of size " << size << " by " << size << ".." << endl;
//    cout << "    Num states: " << fmdp.getNumStates() << endl;
//    cout << "    Num actions: " << fmdp.getNumActions() << endl;

//    cout << " Features" << endl;

//    displayStateColorFeatures(stateFeatures, 5, 5, numFeatures);

//    cout << "\n-- True Rewards --" << endl;
//    fmdp.displayRewards();

//    //solve for the optimal policy
//    vector<unsigned int> opt_policy (fmdp.getNumStates());
//    fmdp.valueIteration(0.001);
//    cout << "-- value function ==" << endl;
//    fmdp.displayValues();
//    fmdp.deterministicPolicyIteration(opt_policy);
//    cout << "-- optimal policy --" << endl;
//    fmdp.displayPolicy(opt_policy);
//    fmdp.calculateQValues();
//    //cout << " Q values" << endl;
//    //fmdp.displayQValues();
//    cout << "state expected feature counts of optimal policy" << endl;
//    double eps = 0.001;
//    double** stateFeatureCnts = calculateStateExpectedFeatureCounts(opt_policy, &fmdp, eps);
//    for(unsigned int s = 0; s < numStates; s++)
//    {
//        double* fcount = stateFeatureCnts[s];
//        cout << "State " << s << ": ";
//        for(unsigned int f = 0; f < numFeatures; f++)
//            cout << fcount[f] << "\t";
//        cout << endl;
//    }
//    
//    cout << "calculate expected feature counts over initial states" << endl;
//    double* expFeatureCnts = calculateExpectedFeatureCounts(opt_policy, &fmdp, eps);
//    for(unsigned int f = 0; f < numFeatures; f++)
//        cout << expFeatureCnts[f] << "\t";
//    cout << endl;


//    //test out empirical estimate of features for demonstrations
//    int trajLength = 25;
//    vector<vector<pair<unsigned int,unsigned int> > > trajectories;
//    for(unsigned int s0 : demoStates)
//    {
//       cout << "demo from " << s0 << endl;
//       vector<pair<unsigned int, unsigned int>> traj = fmdp.monte_carlo_argmax_rollout(s0, trajLength);
//       //for(pair<unsigned int, unsigned int> p : traj)
//           //cout << "(" <<  p.first << "," << p.second << ")" << endl;
//       trajectories.push_back(traj);
//    }
//    double* demoFcounts = calculateEmpiricalExpectedFeatureCounts(trajectories, &fmdp);
//    cout << "Demo f counts" << endl;
//    for(unsigned int f = 0; f < numFeatures; f++)
//        cout << demoFcounts[f] << "\t";
//    cout << endl;

//    cout << "WFCB" << endl;
//    double wfcb = calculateWorstCaseFeatureCountBound(opt_policy, &fmdp, trajectories, eps);
//    cout << wfcb << endl;
//    
//    cout << "True diff" << endl;

//    

//    cout << "Freeing variables" << endl;
//    for(unsigned int s1 = 0; s1 < numStates; s1++)
//    {
//        delete[] stateFeatures[s1];
//        delete[] stateFeatureCnts[s1];
//    }
//    delete[] stateFeatures;
//    delete[] stateFeatureCnts;
//    delete[] demoFcounts;
//    delete[] expFeatureCnts;


}


