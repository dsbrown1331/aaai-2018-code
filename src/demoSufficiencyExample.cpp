#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

//example where VaR performance bound could be used to detect when enough demonstrations
//have been given

using namespace std;


int main() 
{

    ////Experiment parameters
    const unsigned int reps = 20;                    //repetitions per setting
    const vector<unsigned int> numDemos = {1,2,3};            //number of demos to give
    const vector<unsigned int> rolloutLengths = {100};          //max length of each demo
    const vector<double> alphas = {100};                    //confidence param for BIRL
    const unsigned int chain_length = 10000;      //length of MCMC chain
    const int sample_flag = 4;                      //param for mcmc walk type
    const int num_steps = 10;                       //tweaks per step in mcmc
    const bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    const vector<double> steps = {0.01}; //0.01
    const double min_r = -1;
    const double max_r = 1;
    bool removeDuplicates = true;
    bool stochastic = false;


    int startSeed = 1321;
    double eps = 0.001;
    
    //test arrays to get features
    const int numFeatures = 4; //white, red, blue, green
    const int numStates = 12;
    const int width = 3;
    const int height = 4;
    double gamma = 0.95;

    vector<unsigned int> initStates = {11,0,2};
    vector<unsigned int> termStates = {9};
    
    //create directory for results
    string filePath = "./data/demo_sufficiency/";
    string mkdirFilePath = "mkdir -p " + filePath;
    system(mkdirFilePath.c_str());

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
                string filename = "Demo_sufficiency_numdemos" +  to_string(numDemo) 
                                + "_alpha" + to_string((int)alpha) 
                                + "_chain" + to_string(chain_length) 
                                + "_step" + to_string(step)
                                + "_L1sampleflag" + to_string(sample_flag) 
                                + "_rolloutLength" + to_string(rolloutLength)
                                + "_stochastic" + to_string(stochastic)
                                + "_rep" + to_string(rep)+ ".txt";
                cout << filename << endl; 
                ofstream outfile(filePath + filename);
            
                srand(startSeed + 31*rep);
                cout << "------Rep: " << rep << "------" << endl;

                //create random world //TODO delete it when done
                double** stateFeatures = enoughToyWorld(width, height, numFeatures, numStates);
        
                vector<pair<unsigned int,unsigned int> > good_demos;
                vector<vector<pair<unsigned int,unsigned int> > > trajectories; //used for feature counts
              
                /// "true" reward function
                double* featureWeights = new double[5];
                featureWeights[0] = 0;      //white
                featureWeights[1] = -0.5;   //red
                featureWeights[2] = 0;      //blue
                featureWeights[3] = 0.5;    //green
                    
                FeatureGridMDP fmdp(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
                delete[] featureWeights;
                //cout << "Transition function" << endl;
                //fmdp.displayTransitions();
                //cout << "-- Reward function --" << endl;
                //fmdp.displayRewards();
                ///  solve mdp for weights and get optimal policyLoss
                vector<unsigned int> opt_policy (fmdp.getNumStates());
                fmdp.valueIteration(eps);
                //cout << "-- value function ==" << endl;
                //fmdp.displayValues();
                cout << "features" << endl;
                displayStateColorFeatures(stateFeatures, width, height, numFeatures);
                //fmdp.deterministicPolicyIteration(opt_policy);
                fmdp.calculateQValues();
                fmdp.getOptimalPolicy(opt_policy);
                cout << "-- optimal policy --" << endl;
                fmdp.displayPolicy(opt_policy);
                
                cout << "-- feature weights --" << endl;
                fmdp.displayFeatureWeights();

                ///  generate numDemo demos from the initial state distribution
                trajectories.clear(); //used for feature counts
                for(unsigned int d = 0; d < numDemo; d++)
                {
                   int demo_idx = d % initStates.size();
                   unsigned int s0 = initStates[demo_idx];
                   cout << "demo from " << s0 << endl;
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
                {
                    for(pair<unsigned int, unsigned int> p : traj)
                        if(removeDuplicates)
                        {
                            if(std::find(good_demos.begin(), good_demos.end(), p) == good_demos.end())
                                good_demos.push_back(p);
                        }
                        else
                        {    
                            //Remove terminal states from demos for BIRL
                            if(!fmdp.isTerminalState(p.first)) 
                                good_demos.push_back(p);
                        }
                }


                ///  run BIRL to get chain and Map policyLoss ///
                //give it a copy of mdp to initialize
                FeatureBIRL birl(&fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
                birl.addPositiveDemos(good_demos);
                //birl.displayDemos();
                birl.run(eps);
                FeatureGridMDP* mapMDP = birl.getMAPmdp();
                mapMDP->displayFeatureWeights();
                cout << "Recovered reward" << endl;
                mapMDP->displayRewards();

                //solve for the optimal policy
                vector<unsigned int> eval_pi (mapMDP->getNumStates());
                mapMDP->valueIteration(eps);
                mapMDP->calculateQValues();
                mapMDP->getOptimalPolicy(eval_pi);
//                cout << "-- value function ==" << endl;
//                mapMDP->displayValues();
                cout << "-- optimal policy --" << endl;
                mapMDP->displayPolicy(eval_pi);
                //cout << "\nPosterior Probability: " << birl.getMAPposterior() << endl;
                //double base_loss = policyLoss(eval_pi, &fmdp);
                //cout << "Current policy loss: " << base_loss << "%" << endl;

                /// We use the Map Policy as the evaluation policy

                
                
                //write true return,  actual difference, worst-case, and chain info to file
                outfile << "#true return --- true diff --- wfcb --- mcmc ratios" << endl;
                double expOptReturn = getExpectedReturn(&fmdp);
                cout << "Optimal return: " << expOptReturn << endl;
                outfile << expOptReturn << endl;
                outfile << "---" << endl;

                ///compute actual expected return difference
                double trueDiff = expOptReturn - evaluateExpectedReturn(eval_pi, &fmdp, eps);
                cout << "True difference: " << trueDiff << endl;

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
                    //cout << birl.calculateMaxEntPosterior((FeatureGridMDP*)sampleMDP) << endl;
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

}


