#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/optimalTeaching.hpp"
#include "../include/maxent_feature_birl.hpp"
#include "../include/confidence_bounds.hpp"

using namespace std;


vector<unsigned int> getMAPBirlPolicy(FeatureGridMDP *fmdp, vector<pair<unsigned int,unsigned int> > demos, double precision)
{

    double alpha = 100; //50                    //confidence param for BIRL
    unsigned int chain_length = 10000;//1000;//5000;        //length of MCMC chain
    int sample_flag = 4;                      //param for mcmc walk type
    int num_steps = 10;                       //tweaks per step in mcmc
    bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    double step = 0.005; //0.01
    const double min_r = -1;
    const double max_r = 1;

    FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
    birl.addPositiveDemos(demos);
    birl.displayDemos();
    birl.run();
    FeatureGridMDP* mapMDP = birl.getMAPmdp();
    mapMDP->displayFeatureWeights();
    //cout << "Recovered reward" << endl;
    //mapMDP->displayRewards();

    //solve for the optimal policy
    vector<unsigned int> eval_pi (mapMDP->getNumStates());
    mapMDP->valueIteration(precision);
    cout <<"--wall states--" << endl;
    for(unsigned int s = 0; s < mapMDP->getNumStates(); s++)
        cout << mapMDP->isWallState(s) << ",";
    cout << endl;
    cout << "-- value function ==" << endl;
    mapMDP->displayValues();
    mapMDP->calculateQValues();
    mapMDP->getOptimalPolicy(eval_pi);
    cout << "Recovered reward" << endl;
    mapMDP->displayRewards();
    cout << "-- optimal policy --" << endl;
    mapMDP->displayPolicy(eval_pi);
    
    return eval_pi;
}

vector<pair<unsigned int, unsigned int> > processTrajectories(vector<vector<pair<unsigned int,unsigned int> > > trajectories, bool removeDuplicates)
{
    vector<pair<unsigned int,unsigned int> >  demos;
    for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
    {
        for(pair<unsigned int, unsigned int> p : traj)
        {
            if(removeDuplicates)
            {
                if(std::find(demos.begin(), demos.end(), p) == demos.end())
                    demos.push_back(p);
            }
            else
            {    
                demos.push_back(p);
            }
        }
    }
    return demos;
}


int main() 
{


    
    int seed = 1111;

    srand(seed + 13);
    //srand(123);
    //test arrays to get features
    int K = 10;  //TODO change this for stochastic domains!!!!
    int trajLength = 1;
    int numSamples = 100000;
    double precision = 0.0001;
    double eps = 0.0; //2^3
    bool removeDuplicates = true;


 //   FeatureGridMDP* task = generateStochasticDebugMDP();
 //   FeatureGridMDP* task = generateCakmakTask3();
        
//    FeatureGridMDP* task = generateStochasticDebugMDP();
//    FeatureGridMDP* task = generateCakmakTask4();
//    FeatureGridMDP* task = generateCakmakTask1bMAPReward();
     FeatureGridMDP* task = generateRandom9x9World();
    //FeatureGridMDP* task = generateCakmakBadExample3();
    
    
    task->valueIteration(0.0001);
    task->calculateQValues();
    vector< vector<double> > opt_policy = task->getOptimalStochasticPolicy();
    vector<unsigned int> det_pi (task->getNumStates());
    task->getOptimalPolicy(det_pi);
    cout << "-- optimal deterministic policy --" << endl;
    task->displayPolicy(det_pi);
    cout << "-- reward function --" << endl;
    task->displayRewards();
    cout << "-- value function ==" << endl;
    task->displayValues();
    cout << "-- optimal policy --" << endl;
    for(int s=0; s < task->getNumStates(); s++)
    {
        if(!task->isWallState(s))
        {
            cout << "state " << s << ": ";
            for(int a = 0; a < task->getNumActions(); a++)
                cout << opt_policy[s][a] << ",";
            cout << endl;
        }
    }
    
    
    
  /*  //get Cakmak solution

    vector<vector<pair<unsigned int,unsigned int> > > trajectories_cak = OptimalTeachingIRL_Stochastic(task, K, trajLength, numSamples, eps);
    
    cout << "---- Solution ------" << endl;
    int count_cak = 0;
    //print out demos
    for(vector<pair<unsigned int, unsigned int> > traj : trajectories_cak)
    {
        cout << "demo " << count_cak << endl;
        count_cak++;
        for(pair<unsigned int, unsigned int> p : traj)
            cout << "(" <<  p.first << "," << p.second << "), ";
        cout << endl;
    }*/
    
    


    //get set cover solution
    cout << "====================" << endl;
    cout << "deterministic " << endl;
    vector<vector<pair<unsigned int, unsigned int> > > trajectories = solveSetCoverOptimalTeaching(task, K, trajLength);       
     cout << "----- Solution ------" << endl;
    int count = 0;
    //print out demos
    for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
    {
        cout << "demo " << count << endl;
        count++;
        for(pair<unsigned int, unsigned int> p : traj)
            cout << "(" <<  p.first << "," << p.second << "), ";
        cout << endl;
    }
    
    
    
    
//    //get set cover solution
//    cout << "====================" << endl;
//    cout << "Baseline" << endl;
//    opt_demos = solveSetCoverBaseline(task, K, trajLength);       
//     cout << "----- Solution ------" << endl;
//    count = 0;
//    //print out demos
//    for(vector<pair<unsigned int, unsigned int> > traj : opt_demos)
//    {
//        cout << "demo " << count << endl;
//        count++;
//        for(pair<unsigned int, unsigned int> p : traj)
//            cout << "(" <<  p.first << "," << p.second << "), ";
//        cout << endl;
//    }


//learn policy from demonstrations
//       vector<pair<unsigned int, unsigned int> > demos = processTrajectories(trajectories, removeDuplicates);
//       vector<unsigned int> policy = getMAPBirlPolicy(task, demos, precision);
//       double loss = getPolicyLoss(task, policy, precision);
//       double zero_one_loss = policyLoss(policy, task);
//       int num_trajectories = trajectories.size();
//       int num_sa_pairs = demos.size();
//       cout << "Num trajectories = " << num_trajectories << endl;
//       cout << "Num unique state-action pairs = " << num_sa_pairs << endl;
//	    cout << " policy loss = " << loss << endl;
//	    cout << " 0/1 loss = " << zero_one_loss << endl;


}


