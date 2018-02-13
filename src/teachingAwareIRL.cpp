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
    unsigned int chain_length = 200000;//1000;//5000;        //length of MCMC chain
    int sample_flag = 4;                      //param for mcmc walk type
    int num_steps = 10;                       //tweaks per step in mcmc
    bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    double step = 0.05; //0.01
    const double min_r = -1;
    const double max_r = 1;
    int posterior_flag = 0;

    FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps, posterior_flag);
    birl.addPositiveDemos(demos);
    birl.displayDemos();
    birl.run();
    //FeatureGridMDP* mapMDP = birl.getMAPmdp();
    FeatureGridMDP* mapMDP = birl.getMeanMDP(100,10);
    cout << "recovered weights" << endl;
    mapMDP->displayFeatureWeights();
    cout << "posterior =  " << birl.getMAPposterior() << endl;
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

    //1512747944 using this with stochastic debug gives an infinite loop using sol1
    int seed = time(NULL);
    cout << "seed = " << seed << endl;  
  
    srand(seed);
    //srand(123);
    //test arrays to get features
    int K = 5;  //TODO change this for stochastic domains!!!!
    int trajLength = 1;
    int numSamples = 1000;
    double precision = 0.0001;
    double eps = 0.0; //2^3
    bool removeDuplicates = true;


 //   FeatureGridMDP* task = generateStochasticDebugMDP();
    FeatureGridMDP* task = generateCakmakTask3();
 //      FeatureGridMDP* task = generateCakmakTask3e();
        
//    FeatureGridMDP* task = generateStochasticDebugMDP();
//    FeatureGridMDP* task = generateCakmakTask4();
//    FeatureGridMDP* task = generateCakmakTask1bMAPReward();
//    FeatureGridMDP* task = generateRandom9x9World();
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
    
    
    
    



    //learn policy from single optimal info demo

    vector<pair<unsigned int, unsigned int> > demos = processTrajectories(trajectories, removeDuplicates);
    //demos.push_back(make_pair(6,3));

    //demos.push_back(make_pair(1,3));
    //demos.push_back(make_pair(2,3));
    // demos.push_back(make_pair(4,0));
    // demos.push_back(make_pair(0,0));
    
    // demos.push_back(make_pair(15,3));
    // demos.push_back(make_pair(16,0));
    // demos.push_back(make_pair(17,0));
    // demos.push_back(make_pair(11,0));

    // demos.push_back(make_pair(2,1));
    // demos.push_back(make_pair(3,1)); 
    // demos.push_back(make_pair(9,1));
    // demos.push_back(make_pair(15,3));
    // demos.push_back(make_pair(16,3));
    // demos.push_back(make_pair(17,0));
    // demos.push_back(make_pair(11,0));
    // demos.push_back(make_pair(5,0));
     

    
    vector<unsigned int> policy = getMAPBirlPolicy(task, demos, precision);
    double loss = getPolicyLoss(task, policy, precision);
    double zero_one_loss = policyLoss(policy, task);
    cout << " policy loss = " << loss << endl;
    cout << " 0/1 loss = " << zero_one_loss << endl;


}


