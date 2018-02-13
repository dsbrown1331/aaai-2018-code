#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/optimalTeaching.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/feature_birl.hpp"
#include "../include/feature_gain.hpp"

#define SIZE 8

#define FEATURES 8

#define CHAIN_LENGTH 20000
#define PATH_LENGTH 1

#define INTERACTIONS 1
#define STEP_SIZE 0.005
#define ALPHA 100
#define DISCOUNT 0.95

///test out whether my method does better than feature counts
///use BIRL MAP solution as eval policy

using namespace std;

//Added stochastic transitions
////trying large scale experiment for feasible goal rewards
///trying with any random weights and no terminal state
///rewards that don't allow trajectories to the goal.
///using random world and random reward each time


//experiment7_1 has 200 reps, stochastic no duplicates steps 0.01 and 0.05 for 1:9 demos every 2, rollout 200, chain 10000

///TODO I realized that the feasibility is wrt to the number of demos, we should really
///try all possible demos so each run is equivalent, otherwise we'll have different rewards for different numbers of demos and some might be easier/harder and we wont get an apples to apples comparison...

//enum World {SIMPLE, MAZE, CAKMAK1};



	template<typename A, typename B>
pair<B,A> flip_pair(const pair<A,B> &p)
{
	return pair<B,A> (p.second, p.first);
}

	template<typename A, typename B>
multimap<B,A> flip_map(const map<A,B> &src)
{
	multimap<B,A> dst;
	transform(src.begin(), src.end(), inserter(dst, dst.begin()), flip_pair<A,B>);
	return dst;
}

template <typename T1, typename T2>
struct less_second {
	typedef pair<T1, T2> type;
	bool operator ()(type const& a, type const& b) const {
		return a.second < b.second;
	}
};


FeatureGridMDP* makeWorld()
{
	FeatureGridMDP* fmdp = nullptr;

	const int numFeatures = FEATURES; //white, red, blue, green
	const int numStates = SIZE * SIZE;
	const int width = SIZE ;
	const int height = SIZE ;
	double gamma = DISCOUNT;
	vector<unsigned int> initStates;
	for(int i=0;i<numStates;i++)
		initStates.push_back(i);
	vector<unsigned int> termStates = {};
	bool stochastic = false;
	//create random world 
	//double** stateFeatures = initFeaturesSimpleFeatureDomain5x5(numStates, numFeatures);   
	double** stateFeatures = initRandomFeaturesRandomDomain(numStates, numFeatures);
	
	double featureWeights[numFeatures];

	double total_w = 0.0;
	for(unsigned int fi=0; fi<numFeatures; fi++)
	{
		featureWeights[fi] = pow(-1,rand())*(rand()%100)/100;
                total_w += abs(featureWeights[fi]);
	}
	cout << "True feature weights: " ;
	for(unsigned int fi=0; fi<numFeatures; fi++)
	{
		featureWeights[fi] /= total_w; 
		cout << featureWeights[fi] << ", ";
	}
	cout << endl;
	fmdp = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
	return fmdp;

}

double policyLoss(vector<unsigned int> policy,  FeatureGridMDP * mdp, bool count=false)
{
	if (count)
	{
		unsigned int count = 0;
		mdp -> calculateQValues();

		for(unsigned int i=0; i < policy.size(); i++)
		{
			if(! mdp->isOptimalAction(i,policy[i])) {
				//cout <<"incorect?" << i << " " << policy[i] << endl;
				count++;
			}
		} 
		return (double)count/(double)policy.size()*100; 
	}
	else
	{
		double diff =  abs(getExpectedReturn(mdp) - evaluateExpectedReturn(policy, mdp, 0.0001));
		//cout << "Value Diff: " << diff << endl;
		return diff;

	}
}

int main(int argc, char** argv) 
{
	if (argc != 3) {
		cout << "usage: ./test_all algo seed" << endl;
		return 0;
	}
	int seed = atoi(argv[2])*13 + 23;
	string algo = argv[1];
	srand(seed);

string out_file_path = "data/active_bench/" + algo + "_size" + to_string(SIZE) +"_features" + to_string(FEATURES) 
			+ "_mcmcSteps" + to_string(CHAIN_LENGTH) + "_conf" + to_string(ALPHA) + "_trajLength=" + to_string(PATH_LENGTH) + 
"_seed" + argv[2] + ".txt";
    cout << out_file_path << endl;
    ofstream outfile(out_file_path);
	
	double VaR = 0.95;

	//unsigned int rolloutLength = 100;           //max length of each demo
	double alpha = ALPHA; //50                    //confidence param for BIRL
	const unsigned int chain_length = CHAIN_LENGTH;    //1000;//5000;        //length of MCMC chain
	const int sample_flag = 4;                  //param for mcmc walk type
	const int num_steps = 10;                   //tweaks per step in mcmc
	const bool mcmc_reject_flag = true;         //allow for rejects and keep old weights
	double step = STEP_SIZE; //0.01
	const double min_r = -1; //not used
	const double max_r = 1;  //not used
	bool removeDuplicates = true;
	double eps = 0.0001;

	FeatureGridMDP* fmdp = generateRandomWorldNStates(FEATURES, SIZE);//makeWorld();
	unsigned int numStates = fmdp->getNumStates();

	vector<unsigned int> initStates;
	for(unsigned int s = 0; s < fmdp->getNumStates(); s++)
		if(fmdp->isInitialState(s))
			initStates.push_back(s);

	///  solve mdp for weights and get optimal policyLoss
	vector<unsigned int> opt_policy (fmdp->getNumStates());
	fmdp->valueIteration(eps);
	
	cout << "-- feature weights --" << endl;
	fmdp->displayFeatureWeights();

	cout << "-- rewards --" << endl;
	fmdp->displayRewards();

	fmdp->calculateQValues();
	
	fmdp->getOptimalPolicy(opt_policy);
	cout << "-- optimal policy --" << endl;
	fmdp->displayPolicy(opt_policy);

	///  generate demo
	vector<pair<unsigned int,unsigned int> > good_demos;
	unsigned int s = rand()% fmdp->getNumStates();
	for( unsigned int i = 0; i < PATH_LENGTH; i++)
	{
		good_demos.push_back(make_pair(s,opt_policy[s]));
		unsigned int ns = fmdp->getNextState(s,opt_policy[s]);  
		if(ns == s) s = rand()% fmdp->getNumStates();
		else s = ns;
	}	

	
    

	for(unsigned int itr = 0; itr < INTERACTIONS; itr++)
	{
	    cout << "itr " << itr << endl;
	    cout << "size of demos = " << good_demos.size() << endl;
	    ///  run BIRL to get chain and Map policyLoss ///
	    //give it a copy of mdp to initialize
	    FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps); 
	    birl.addPositiveDemos(good_demos);

	    //FeatureInfoGainCalculator gain_computer = FeatureInfoGainCalculator(&birl);

	    birl.displayDemos();
	    birl.run();
	    cout << "birl ran" << endl;	

	    FeatureGridMDP* mapMDP = birl.getMAPmdp();
	    mapMDP->displayFeatureWeights();
	    cout << "Recovered reward" << endl;
	    mapMDP->displayRewards();

	    vector<unsigned int> map_policy  (mapMDP->getNumStates());
	    mapMDP->valueIteration(eps);
	    mapMDP->deterministicPolicyIteration(map_policy);
	    mapMDP->displayPolicy(map_policy);
	    double evd_loss = policyLoss(map_policy, fmdp);
	    double zero_one_loss = policyLoss(map_policy, fmdp, true);
        
        
	    cout << "Current policy evd loss: "  << evd_loss << endl;
	    cout << "0-1 loss: " << zero_one_loss << endl;
	    outfile << good_demos.size() << "," << evd_loss << "," << zero_one_loss << endl;
	
	
	
        cout << "iter: " << itr << endl;
        pair<unsigned int, unsigned int> best_pair;
        if(algo == "var")
        {
		    // --- Active VaR -----
		    cout << "------------"<< endl;

		    //Get V^* and V^\pi_eval for each start state 
		    vector<vector<double>> evds(initStates.size(), vector<double>(chain_length));
		    for(unsigned int i=0; i<chain_length; i++)
		    {
			    GridMDP* sampleMDP = (*(birl.getRewardChain() + i));
			    vector<unsigned int> sample_pi(sampleMDP->getNumStates());
			    sampleMDP->getOptimalPolicy(sample_pi);
			    vector<double> Vstar_vec =  getExpectedReturnVector(sampleMDP);
			    vector<double> Vhat_vec = evaluateExpectedReturnVector(map_policy, sampleMDP, eps);
			    for(unsigned int j = 0; j < Vstar_vec.size(); j++)
			    {
				    double EVDiff = Vstar_vec[j] - Vhat_vec[j];
				    evds[j][i] = EVDiff;
			    }
		    }    
		    //output VaR data
		    //ofstream outfile_var("data/active/var_" + filename);
		    cout << "VaR:" << endl;
		    cout.precision(4);
		    unsigned int query_state = 0;
		    double max_VaR = 0;
		    for(unsigned int s = 0; s < evds.size(); s++)
		    {
			    std::sort(evds[s].begin(), evds[s].end());
			    int VaR_index = (int) chain_length * VaR;
			    double eval_VaR = evds[s][VaR_index];  
			    if (eval_VaR > max_VaR)
			    {
				    max_VaR = eval_VaR;
				    query_state = s;
			    }   
			    if(s % fmdp->getGridWidth() < fmdp->getGridWidth() - 1)
			    {  
				    //outfile_var << eval_VaR << ",";
				    cout << eval_VaR << ",";
			    }else{
				    //outfile_var << eval_VaR << endl;
				    cout << eval_VaR << "," << endl;
			    }
		    }
		    //outfile_var.close();
		    cout << endl << "VaR query: " ;   
		    cout << query_state << endl;
		    best_pair = make_pair(query_state, opt_policy[query_state]);
		    good_demos.push_back(best_pair); 
		    cout << "size of demos = " << good_demos.size() << endl;
		    cout << "------------"<< endl;
        }
        else if(algo == "entropy")
        {
          	FeatureInfoGainCalculator gain_computer = FeatureInfoGainCalculator(&birl);

		    // --- Active Entropy -----
		    unsigned int numStates = fmdp->getNumStates();
		    map<unsigned int, double> entropies;
		    cout << "Entropies: " << endl ;   
		    for(unsigned int s = 0; s < numStates; s+= 1)
		    {
			    pair<unsigned int,unsigned int> state_action;
			    unsigned int a;
			    entropies.insert(make_pair(s,0.0));
			    for( unsigned int a = 0 ; a < fmdp->getNumActions(); a++) //
			    {
				    state_action = make_pair(s,a);
				    double ent = gain_computer.getEntropy(state_action); 
				    entropies[s] += ent;  
				    cout.precision(15);
				    //cout <<" ----> Entropy for action " << a << ": " << ent << endl;   
			    }
			    cout.precision(5);
			    entropies[s] /= fmdp->getNumActions();
			    cout  << entropies[s] << ", " ;
			    if (s % SIZE == SIZE - 1) cout << endl;
		    }
		    cout << endl;
		    vector<pair<unsigned int, double> > argmax_entropies(entropies.begin(), entropies.end());
		    sort(argmax_entropies.begin(), argmax_entropies.end(), less_second<unsigned int, double>());
		    cout << "Entropy query: " ;          
		    for ( unsigned int idx = 0; idx < PATH_LENGTH ; ++idx)
		    {
			    cout << argmax_entropies[numStates-1-idx].first;
			    unsigned int list_idx = idx;
			    do{
				    best_pair = make_pair(argmax_entropies[numStates-1-list_idx].first ,opt_policy[argmax_entropies[numStates-1-list_idx].first]);
				    list_idx++;
			    }while(birl.isDemonstration(best_pair));	
			    //if(!birl1.isDemonstration(best_pair))	
			    good_demos.push_back(best_pair); 

		    }
		    cout << endl << "------------"<< endl;
        }
        else if(algo == "random")
        {
		    //random
		    cout << "Random query: " ;          
		    for ( unsigned int idx = 0; idx < PATH_LENGTH ; ++idx)
		    {
		        int rand_s = rand() % numStates;
		        cout << rand_s;
		        best_pair = make_pair(rand_s ,opt_policy[rand_s]);	
		        good_demos.push_back(best_pair); 

		    }
		    cout << endl;
	    }
	    cout << "end of loop" << endl;
	}
    FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps); 
    birl.addPositiveDemos(good_demos);


    birl.displayDemos();
    birl.run();
    cout << "birl ran" << endl;



    FeatureGridMDP* mapMDP = birl.getMAPmdp();
    mapMDP->displayFeatureWeights();
    cout << "Recovered reward" << endl;
    mapMDP->displayRewards();

    vector<unsigned int> map_policy  (mapMDP->getNumStates());
    mapMDP->valueIteration(eps);
    mapMDP->deterministicPolicyIteration(map_policy);
    mapMDP->displayPolicy(map_policy);
    double evd_loss = policyLoss(map_policy, fmdp);
    double zero_one_loss = policyLoss(map_policy, fmdp, true);
    
    cout << "Current policy loss: "  << evd_loss << "%" << endl;
    cout << "Current 0-1 policy loss: "  << zero_one_loss << "%" << endl;
    outfile << good_demos.size() << "," << evd_loss << "," << zero_one_loss << endl;
	
	

    
	//clean up
	double** stateFeatures = fmdp->getStateFeatures();
	//delete features
	for(unsigned int s1 = 0; s1 < fmdp->getNumStates(); s1++)
	{
		delete[] stateFeatures[s1];
	}
	delete[] stateFeatures;

	delete fmdp;


}


