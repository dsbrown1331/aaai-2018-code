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

//TODO: get it to report Policy loss
//Get it to run multiple iterations and write to file for later averaging

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


int main(int argc, char *argv[]) 
{

    if(argc < 4)
    {
        cout << "Usage: ./cakmakTasksExperiment algo_name task trajLength" << endl;
        exit(EXIT_FAILURE);
    }
    string algo = argv[1];
    string world = argv[2]; //TODO change this programatically
    string traj_length = argv[3];
    
    int seed = 1234;

    
    //srand(123);
    //test arrays to get features
    int K = 6;  //TODO change this for stochastic domains!!!!
    int trajLength = stoi(traj_length);
    int numSamples = 10000000;
    double eps = 0; //2^3
    double precision = 0.0001;
    bool removeDuplicates = true;
    int replicates = 20;
    
    string params = "";
    if(algo == "cakmak")
        params = to_string(numSamples);
    string out_file_path = "data/machine_teaching/" + algo + "_" + params + "_" + world + "_" + "trajLength=" + traj_length + ".txt";
    cout << out_file_path << endl;
    ofstream outfile(out_file_path);
    
    for(int rep = 0; rep < replicates; rep++)
    {
        srand(seed + 13 * rep);


//    FeatureGridMDP* task = generateStochasticDebugMDP();
    FeatureGridMDP* task = NULL;
    if(world == "task1")
        task = generateCakmakTask1();
    else if(world == "task2")
        task = generateCakmakTask2();
    else if(world == "task3")
        task = generateCakmakTask3();
    else if(world == "task4")
        task = generateCakmakTask4();
    else if(world == "random")
        task = generateRandom9x9World();
    else
    {
        cout << "NOT VALID TASK" << endl;
        cout << "Usage: ./cakmakTasksExperiment algo_name task" << endl;
        exit(EXIT_FAILURE);
    }
        
    //FeatureGridMDP* task = generateCakmakBadExample3();
    
    
    task->valueIteration(precision);
    task->calculateQValues();
    vector<unsigned int> eval_pi (task->getNumStates());
    task->getOptimalPolicy(eval_pi);
    cout <<"--wall states--" << endl;
    bool* wallStates = task->getWallStates();
    for(unsigned int s = 0; s < task->getNumStates(); s++)
        cout << wallStates[s] << ",";
    cout << endl;
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
    
    

        cout << "========== REP " << rep << "==============" << endl;
        cout << "World = " << world << endl;

        clock_t c_start = clock();
       
        vector<vector<pair<unsigned int,unsigned int> > > trajectories;
        
                    
        if(algo == "old_cakmak")
        {
            //Cakmak algorithm
            cout << "using old Cakmak algorithm" << endl;
            trajectories = OptimalTeachingIRL_Stochastic(task, K, trajLength, numSamples, eps);
        }
        else if(algo == "cakmak")
        {
            //vectorized Cakmak algorithm
            cout << "using vectorized cakmak alg" << endl;
            trajectories = OptimalTeachingIRL_Stochastic_Vectorized(task, K, trajLength, numSamples, eps);
        }
        else if(algo == "baseline")
        {
            //set-cover baseline
            cout << "using baseline algorithm" << endl;
            trajectories = solveSetCoverBaseline(task, K, trajLength);
        }
        else if(algo == "non-redundant")
        {
            //set-cover randomized without redundancies
            cout << "using random no redundancies algorithm" << endl;
            trajectories= solveSetCoverOptimalTeaching_sol1(task, K, trajLength); 
        }
        else if(algo == "randomized")
        {
                //set-cover randomized without redundancies multiple iterations
            cout << "multiple random iterations using random no redundancies algorithm" << endl;
            int cnt = 0;
            vector<vector<pair<unsigned int,unsigned int> > > trial;
            unsigned int min_num = 100000;
            while(cnt < 20)
            {
                cnt++;
                trial = solveSetCoverOptimalTeaching_sol1(task, K, trajLength); 
                if(trial.size() < min_num)
                {
                    min_num = trial.size();
                    cout << "found solution with " << min_num << " trajectories" << endl;
                    trajectories = trial;
                }
            }
        }
        else if(algo == "random_samples")
        {
            //random disjoint samples, I chose 18 to compare with NRCC
            for(int i=0; i<20; i++)
            {
                vector<pair<unsigned int,unsigned int> > demo;
                for ( unsigned int idx = 0; idx < trajLength; ++idx)
		        {
		            int rand_s = rand() % task->getNumStates();
		            pair<unsigned int, unsigned int> best_pair = make_pair(rand_s ,eval_pi[rand_s]);	
   		            cout << "(" << rand_s << "," << best_pair.second << "), ";
		            demo.push_back(best_pair); 
		            

		        }
		        cout << endl;
		        trajectories.push_back(demo);
		    }
		    
        }
        
        clock_t c_end = clock();

       //learn policy from demonstrations
       vector<pair<unsigned int, unsigned int> > demos = processTrajectories(trajectories, removeDuplicates);
       vector<unsigned int> policy = getMAPBirlPolicy(task, demos, precision);
       double loss = getPolicyLoss(task, policy, precision);
       double zero_one_loss = policyLoss(policy, task);
       double time_elapsed = (c_end-c_start)*1.0 / CLOCKS_PER_SEC;
       int num_trajectories = trajectories.size();
       int num_sa_pairs = demos.size();
       cout << "\n[Timing] Time passed: " << time_elapsed << " s\n"; 
       cout << "Num trajectories = " << num_trajectories << endl;
       cout << "Num unique state-action pairs = " << num_sa_pairs << endl;
	    cout << " policy loss = " << loss << endl;
	    cout << " 0/1 loss = " << zero_one_loss << endl;
	
	    outfile << time_elapsed << "," << num_trajectories << "," << loss << "," << zero_one_loss << "," << num_sa_pairs << endl;
	
        cout << "---- Solution ------" << endl;
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
        
    }
    outfile.close();


//    //get set cover solution
//    cout << "====================" << endl;
//    cout << "randomized" << endl;
//    c_start = clock();
//    vector<vector<pair<unsigned int, unsigned int> > > opt_demos = solveSetCoverOptimalTeaching_sol1(task, K, trajLength);    
//       c_end = clock();
//   cout << "\n[Timing] Time passed: " << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n"; 
//   cout << "Num demos = " << demos.size() << endl;   
//     cout << "----- Solution ------" << endl;
//    int count = 0;
//    //print out demos
//    for(vector<pair<unsigned int, unsigned int> > traj : opt_demos)
//    {
//        cout << "demo " << count << endl;
//        count++;
//        for(pair<unsigned int, unsigned int> p : traj)
//            cout << "(" <<  p.first << "," << p.second << "), ";
//        cout << endl;
//    }
//    
//    
//    //get set cover solution
//    cout << "====================" << endl;
//    cout << "Baseline" << endl;
//    c_start = clock();
//    opt_demos = solveSetCoverBaseline(task, K, trajLength);       
//    c_end = clock();
//   cout << "\n[Timing] Time passed: " << (c_end-c_start)*1.0 / CLOCKS_PER_SEC << " s\n"; 
//   cout << "Num demos = " << demos.size() << endl;
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


}


