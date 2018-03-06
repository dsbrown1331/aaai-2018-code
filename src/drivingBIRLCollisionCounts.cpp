#include <unordered_map>
#include <string>
#include "../include/q_learner_driving.hpp"
#include "../include/driving_world.hpp"
#include "../include/feature_birl_qlearning.hpp"
#include "../include/confidence_bounds_qlearning.hpp"
#include <algorithm>
#include <fstream>

double evalFeatureWeights[6];
string evalPolicyName;

//script to test out q-learner in BIRL
int main(int argc, char* argv[])
{

    // Check the number of parameters
    if (argc != 2) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << "eval_policy" << std::endl;
        std::cerr << "eval policy can be one of the following: \n"
                        "\t\t\t on_road \n"
                        "\t\t\t right_safe \n"
                        "\t\t\t nasty" << endl;

        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    else
    {
        if(strcmp( argv[1],"on_road") == 0)
        {
            double weights[] = {0,  //collision
                               -0.5,  //offroad left
                                 0,  //road left lane
                                  0,  //road center lane
                                 0,  //road right lane
                               -0.5}; //offroad right
                          
            copy(begin(weights), end(weights), begin(evalFeatureWeights));
            evalPolicyName = "on_road";
        }
        else if(strcmp( argv[1], "right_safe") == 0)
        {


            double weights[] = {-1,  //collision
                               -1,  //offroad left
                                 0,  //road left lane
                                  0.1,  //road center lane
                                 0.5,  //road right lane
                               -1}; //offroad right
            copy(begin(weights), end(weights), begin(evalFeatureWeights));
            evalPolicyName = "right_safe";
        }
        else if(strcmp( argv[1], "nasty") == 0)
        {


            double weights[] = {1,  //collision
                               -1,  //offroad left
                                 0,  //road left lane
                                  0,  //road center lane
                                 0,  //road right lane
                               -1}; //offroad right                      
            copy(begin(weights), end(weights), begin(evalFeatureWeights));
            evalPolicyName = "nasty";
        }
        else if(strcmp( argv[1], "expert")==0)
        {

            double weights[] = {-0.4,  //collision
                                   -0.2,  //offroad left
                                     0,  //road left lane
                                      0,  //road center lane
                                     0,  //road right lane
                                   -0.2}; //offroad right     
            copy(begin(weights), end(weights), begin(evalFeatureWeights));              
            evalPolicyName = "expert";
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << "eval_policy" << std::endl;
            std::cerr << "eval policy can be one of the following: \n"
                            "\t\t\t on_road \n"
                            "\t\t\t right_safe \n"
                            "\t\t\t nasty"<< endl;

            /* "Usage messages" are a conventional way of telling the user
             * how to run a program if they enter the command incorrectly.
             */
            return 1;
        }
        
    
    
    }
    bool debug = false;
    int mc_rolloutLength = 100;
    int mc_numRollouts = 200;
    const unsigned int reps = 20;
    int startSeed = 3132;
    //srand(time(NULL));
    //create world
    bool visualize = false;
    int numStateFeatures = 12;
    int numRewardFeatures = 6;
    bool twoCars = false;
    double exploreRate = 0.8; //without a goal, I think qlearning with epsilon should be close to 1 so we see lots of states and take all possible actions many times
    double learningRate = 0.1;
    int numActions = 3;
    double gamma = 0.9;
    int numQSteps = 6000; 
    int demo_length = 100;
    double min_reward = -2;
    double max_reward = 2;
    unsigned int chain_len = 2000; 
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
