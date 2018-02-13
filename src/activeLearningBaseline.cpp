#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/optimalTeaching.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

using namespace std;

#define SIZE 8

#define FEATURES 8

#define CHAIN_LENGTH 20000
#define PATH_LENGTH 1

#define INTERACTIONS 15 
#define STEP_SIZE 0.005
#define ALPHA 100
#define DISCOUNT 0.95

//TODO: get it to report Policy loss
//Get it to run multiple iterations and write to file for later averaging


vector<unsigned int> getMAPBirlPolicy(FeatureGridMDP *fmdp, vector<pair<unsigned int,unsigned int> > demos, double precision)
{

    double alpha = ALPHA; //50                    //confidence param for BIRL
    unsigned int chain_length = CHAIN_LENGTH;//1000;//5000;        //length of MCMC chain
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


int main(int argc, char *argv[]) 
{

    if(argc != 2)
    {
        cout << "usage: ./active_bench seed" << endl;
        return 0;
    }
    int seed = atoi(argv[1])*13 + 23;

    
    srand(seed);
    //test arrays to get features
    int K = 10;  //TODO change this for stochastic domains!!!!
    int trajLength = 1;
    //int numSamples = 100000;
    //double eps = 0; //2^3
    double precision = 0.0001;
    bool removeDuplicates = true;
        cout << "starting" << endl;    
    string params = "";
    string out_file_path = "data/active_bench/opt_size" + to_string(SIZE) +"_features" + to_string(FEATURES) + 
                "_mcmcSteps" + to_string(CHAIN_LENGTH) + "_conf" + to_string(ALPHA) + "_trajLength=" + 
                to_string(PATH_LENGTH) + "_seed" + argv[1] + ".txt";
    cout << out_file_path << endl;
    ofstream outfile(out_file_path);

    FeatureGridMDP* task = NULL;
    //TODO
    task = generateRandomWorldNStates(FEATURES, SIZE);//generateRandom9x9World(); // makeWorld();//
    
    task->valueIteration(precision);
    task->calculateQValues();
    vector<unsigned int> eval_pi (task->getNumStates());
    task->getOptimalPolicy(eval_pi);
    cout << "-- optimal deterministic policy --" << endl;
    task->displayPolicy(eval_pi);
    vector< vector<double> > opt_policy = task->getOptimalStochasticPolicy();
    cout << "-- reward function --" << endl;
    task->displayRewards();
    cout << "-- value function ==" << endl;
    task->displayValues();
//    cout << "-- optimal stochastic policy --" << endl;
//    for(unsigned int s=0; s < task->getNumStates(); s++)
//    {
//        if(!task->isWallState(s))
//        {
//            cout << "state " << s << ": ";
//            for(unsigned int a = 0; a < task->getNumActions(); a++)
//                cout << opt_policy[s][a] << ",";
//            cout << endl;
//        }
//    }
    
    
    
    
        vector<vector<pair<unsigned int,unsigned int> > > trajectories;
        
        trajectories= solveSetCoverOptimalTeaching_sol1(task, K, trajLength); 
        
        cout << "number of optimal trajectories = " << trajectories.size() << endl;
       
       //learn policy from demonstrations
       vector<pair<unsigned int, unsigned int> > demos = processTrajectories(trajectories, removeDuplicates);
       
       vector<pair<unsigned int, unsigned int> > demonstrations;
       int cnt = 0;
       for(pair<unsigned int, unsigned int> p : demos)
       {
            cout << "iteration " << cnt++ << endl;
           demonstrations.push_back(p);
       
           //print out current demonstrations
            for(pair<unsigned int, unsigned int> p : demonstrations)
            {
                    cout << "(" <<  p.first << "," << p.second << "), ";

            }
            cout << endl;
           //add next pair to demonstration set
           vector<unsigned int> policy = getMAPBirlPolicy(task, demonstrations, precision);
           double loss = getPolicyLoss(task, policy, precision);
           double zero_one_loss = policyLoss(policy, task);
           //double time_elapsed = (c_end-c_start)*1.0 / CLOCKS_PER_SEC;
           int num_sa_pairs = demonstrations.size();
           //cout << "Num trajectories = " << num_trajectories << endl;
           cout << "Num unique state-action pairs = " << num_sa_pairs << endl;
	        cout << " policy loss = " << loss << endl;
	        cout << " 0/1 loss = " << zero_one_loss << endl;
	
	        outfile << num_sa_pairs << "," << loss << "," << zero_one_loss << endl;
	
            cout << "---- Solution ------" << endl;
            int count = 0;
            
            
       
    }
     outfile.close();
    
}


