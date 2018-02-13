#include <unordered_map>
#include <string>
#include "../include/q_learner_driving.hpp"
#include "../include/driving_world.hpp"
#include "../include/feature_birl_qlearning.hpp"
#include "../include/confidence_bounds_qlearning.hpp"
#include <algorithm>
#include <fstream>

//This experiment is to try and show that worst-case feature counts do silly things 

//this is for a naive eval policy

//it uses an evaluation policy defined by evalFeatureWeights
//double evalFeatureWeights[] = {0,  //collision
//                                 0,  //tailgate
//                               -0.5,  //offroad left
//                                 0,  //road left lane
//                                  0,  //road center lane
//                                 0,  //road right lane
//                               -0.5}; //offroad right
//                            // 0.0, //car to left of me  //TODO makes things weird!
//                            // 0.0};//car to right of me

//string evalPolicyName = "stay_on_road";


//double evalFeatureWeights[] = {-1,  //collision
//                                 -1,  //tailgate
//                               -1,  //offroad left
//                                 0,  //road left lane
//                                  0.1,  //road center lane
//                                 0.5,  //road right lane
//                               -1}; //offroad right
//                            // 0.0, //car to left of me  //TODO makes things weird!
//                            // 0.0};//car to right of me
//string evalPolicyName = "right_lane";


double evalFeatureWeights[] = {1,  //collision
                                 1,  //tailgate
                               -1,  //offroad left
                                 0,  //road left lane
                                  0,  //road center lane
                                 0,  //road right lane
                               -1}; //offroad right
                            // 0.0, //car to left of me  //TODO makes things weird!
                            // 0.0};//car to right of me
string evalPolicyName = "nasty";

//double evalFeatureWeights[] = {-0.4,  //collision
//                                 -0.2,  //tailgate
//                               -0.2,  //offroad left
//                                 0,  //road left lane
//                                  0,  //road center lane
//                                 0,  //road right lane
//                               -0.2}; //offroad right
//                            // 0.0, //car to left of me  //TODO makes things weird!
//                            // 0.0};//car to right of me
//string evalPolicyName = "expert";

//script to test out q-learner in BIRL
int main()
{
    bool debug = false;
    int mc_rolloutLength = 100;
    int mc_numRollouts = 200;
    const unsigned int reps = 20;
    int startSeed = 3132;
    //srand(time(NULL));
    //create world
    bool visualize = false;
    int numStateFeatures = 12;
    int numRewardFeatures = 7;
    bool twoCars = false;
    double exploreRate = 0.8; //without a goal, I think qlearning with epsilon should be close to 1 so we see lots of states and take all possible actions many times
    double learningRate = 0.1;
    int numActions = 3;
    double gamma = 0.9;
    int numQSteps = 6000; 
    int demo_length = 100;
    double min_reward = -2;
    double max_reward = 2;
    unsigned int chain_len = 2000; ///TODO change back!
    double mcmc_step = 0.01;
    double alpha_conf = 5;
    int sample_flag = 4;
    bool mcmc_reject = true;
    bool remove_duplicates = true;
    //bool givePolicyDemo = false;

    int num_proposal_steps = 10; //doesn't matter for manifold all walk


    vector<pair<string,unsigned int> > demonstration;
    vector<State> trajectory;
    cout << "RUNNING " << evalPolicyName << " TEST ---" << endl;
    cout << "Initialized world" << endl;
 
  
    double ave_c = 0;
    for(unsigned int rep = 0; rep < reps; rep++)
    {
       
        DrivingWorld eval_world(visualize, evalFeatureWeights, numStateFeatures, numRewardFeatures, twoCars);
     
        TabularQLearner opt_policy(&eval_world, numActions, gamma);
        opt_policy.trainEpoch(numQSteps, exploreRate, learningRate);
        demonstration.clear();  
        trajectory.clear();
        cout << "Generating demonstration" << endl;
        if(debug)
            eval_world.setVisuals(true);
        else
            eval_world.setVisuals(false);
        State initState = eval_world.startNewEpoch();
        trajectory.push_back(initState);
        State state = initState;
        //cout << "init state: " << initState.toString() << endl;
        for(int step = 0; step < demo_length; step++)
        {
            if(debug)
                cout << "============" << step << endl;   
            
            unsigned int action = opt_policy.getArgmaxQvalues(state);
            //save state-action pair in demonstration
            pair<string, unsigned int> sa_demo = make_pair(state.toStateString(), action);
            if(remove_duplicates)
            {
                //check not in demo already
                if (std::find(demonstration.begin(), demonstration.end(), sa_demo) == demonstration.end())
                    demonstration.push_back(sa_demo);
            }
            else
                demonstration.push_back(sa_demo);
            pair<State,double> nextStateReward = eval_world.updateState(action);
            trajectory.push_back(nextStateReward.first);
            State nextState = nextStateReward.first;
            //double reward = nextStateReward.second;
            state = nextState;
        }
        
        //qbot.displayQvalues();

        //count number of collisions
        int c_count = 0;
        for(State s : trajectory)
        {            
            c_count += s.getRewardFeature(0);
        }
        ave_c += c_count;
    }
    ave_c = ave_c / reps;
    cout << "ave num collisions = " << ave_c << endl;

    
        

    return 0;

}
