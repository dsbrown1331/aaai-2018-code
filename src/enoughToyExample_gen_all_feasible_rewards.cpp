#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/maxent_feature_birl.hpp"
#include <fstream>
#include <string>

///generate all feasible rewards that match a demonstration to plot in python

using namespace std;


int main() 
{

    ////Experiment parameters
    const unsigned int reps = 1;                    //repetitions per setting
   
    int startSeed = 1323;
    double eps = 0.00001;
    
    //test arrays to get features
    const int numFeatures = 2; //white, red
    const int numStates = 12;
    const int width = 3;
    const int height = 4;
    double gamma = 0.95;
    vector<unsigned int> initStates = {11};
    vector<unsigned int> termStates = {9};
    bool stochastic = false;

    //set up file for output
    string filename = "demo_from_state11_L1.txt";
    cout << filename << endl; 
    //ofstream outfile("data/enoughIsEnough/experiment_toy_feasible/" + filename);

    //create random world //TODO delete it when done
    double** stateFeatures = feasibleToyWorld(width, height, numFeatures, numStates);

    vector<pair<unsigned int,unsigned int> > good_demos;
    good_demos.push_back(make_pair(11,2));
    good_demos.push_back(make_pair(10,2));



    for(unsigned int rep = 0; rep < reps; rep++)
    {
    
        srand(startSeed + 31*rep);
        cout << "------Rep: " << rep << "------" << endl;

        //sample from L1-ball
        double* featureWeights = sample_unit_L1_norm(numFeatures);
        featureWeights[0] = -0.91116;
        featureWeights[1] = -1 - featureWeights[0];
      
        ///  create a random weight in unit square
//        double* featureWeights = new double[numFeatures];
//        featureWeights[0] = 2.0 * ((double) rand() / RAND_MAX) - 1.0;      //white
//        featureWeights[1] = 2.0 * ((double) rand() / RAND_MAX) - 1.0;   //red
//            

        FeatureGridMDP fmdp(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
        delete[] featureWeights;

        vector<unsigned int> opt_policy (fmdp.getNumStates());
        fmdp.valueIteration(eps);
        cout << "-- value function ==" << endl;
        fmdp.displayValues();
        //cout << "features" << endl;
        //displayStateColorFeatures(stateFeatures, width, height, numFeatures);
        fmdp.calculateQValues();
        fmdp.getOptimalPolicy(opt_policy);
        cout << "-- optimal policy --" << endl;
        fmdp.displayPolicy(opt_policy);
        
        cout << "-- feature weights --" << endl;
        fmdp.displayFeatureWeights();
        
        //check if proposed reward function has optimal policy that matches the demonstration
        
        bool matches = true;
        for(pair<unsigned int, unsigned int> p : good_demos)
        {
            if(!fmdp.isOptimalAction(p.first, p.second))
            {
                matches = false;
                break;
            }
        
        }
        //write to file if a match
        if(matches) 
        {
//            cout << "matches" << endl;
//            outfile << fmdp.getFeatureWeights()[0] << "," << fmdp.getFeatureWeights()[1] << endl; 
        }

        

    }

}

