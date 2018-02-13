#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/abbeel_projection.hpp"

using namespace std;



int main() 
{
    srand(time(NULL));
    //test arrays to get features
    const int numFeatures = 5; //white, red, blue, yellow, green
    const int numStates = 9;
    const int size = 3;
    double featureWeights[] = {-0.1,-0.2,-0.3,-0.4,-0.5};
    //double featureWeights[] = {0,-0.5,+0.5};
    double epsilon = 0.001;  //for abbeel algorithm
    double gamma = 0.99;
    double eps = 0.001;
    int trajLength = 100;

    double** stateFeatures = randomGridNavDomain(numStates, numFeatures);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {8};
    vector<unsigned int> termStates = {6};
    vector<unsigned int> wallStates = {4,7};
    vector<unsigned int> demoStates = initStates;
    bool stochastic = false;
   
    FeatureGridMDP fmdp(size, size, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    cout << "Setting wall states" << endl;
    for(unsigned int s : wallStates)
        fmdp.setWallState(s);

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
    
    cout << "-- Transitions --" << endl;
    fmdp.displayTransitions();
    //cout << " Q values" << endl;
    //fmdp.displayQValues();
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


    //test out empirical estimate of features for demonstrations
    
//    vector<vector<pair<unsigned int,unsigned int> > > trajectories;
//    for(unsigned int s0 : demoStates)
//    {
//       cout << "demo from " << s0 << endl;
//       vector<pair<unsigned int, unsigned int>> traj = fmdp.monte_carlo_argmax_rollout(s0, trajLength);
//       for(pair<unsigned int, unsigned int> p : traj)
//           cout << "(" <<  p.first << "," << p.second << ")" << endl;
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
//    cout << "Abbeel projection algorithm" << endl;
//    ProjectionIRL projectionIRL(&fmdp);
//    vector<unsigned int> projection_policy(fmdp.getNumStates());
//    projectionIRL.getProjectionPolicy(projection_policy, trajectories, epsilon);
//    fmdp.displayPolicy(projection_policy);
//    double* projectionFcounts = calculateExpectedFeatureCounts(projection_policy, &fmdp, eps);
//    cout << "abbeel policy f counts" << endl;
//    for(unsigned int f = 0; f < numFeatures; f++)
//        cout << projectionFcounts[f] << "\t";
//    cout << endl;
//    cout << "L2 difference: " << twoNormDiff(demoFcounts, projectionFcounts, numFeatures) << endl;
//    

//    cout << "Freeing variables" << endl;
//    for(unsigned int s1 = 0; s1 < numStates; s1++)
//    {
//        delete[] stateFeatures[s1];
//        //delete[] stateFeatureCnts[s1];
//    }
//    delete[] stateFeatures;
//    //delete[] stateFeatureCnts;
//    delete[] demoFcounts;
//    delete projectionFcounts;
//    //delete[] expFeatureCnts;


}


