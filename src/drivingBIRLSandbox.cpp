#include <unordered_map>
#include <string>
#include "../include/q_learner_driving.hpp"
#include "../include/driving_world.hpp"
#include "../include/feature_birl_qlearning.hpp"
#include "../include/confidence_bounds_qlearning.hpp"
#include <algorithm>



//DONE need to add features for distance to car in all lanes. I think this will allow the car to prefer one lane and to learn to drive off road and side swipe when cars come by. Currently it tries to prefer one lane until a car comes, then it changes lanes, but then it says "why am I not in the prefered lane?" and comes right back and collides.
//TODO probably need more discrete distances to car so it knows where it is, otherwise it collides, but doesn't know where it hit 

//script to test out q-learner in BIRL
int main()
{
    srand(time(NULL));
    //create world
    bool visualize = false;
    int numFeatures = 9;
    double exploreRate = 0.8; //without a goal, I think qlearning with epsilon should be close to 1 so we see lots of states and take all possible actions many times
    double learningRate = 1.0;
    int numActions = 3;
    double gamma = 0.95;
    int numDemoPolicyQsteps = 10000;
    //int numBIRL_QSteps = 10000;     
    int demo_length = 1000;
    //double min_reward = -2;
    //double max_reward = 2;
    //unsigned int chain_len = 500;
    //double mcmc_step = 0.01;
    //double alpha_conf = 10;
    //int sample_flag = 4;
    //bool mcmc_reject = true;
    bool remove_duplicates = true;
    bool givePolicyDemo = false;
    
    int mc_rolloutLength = 1000;
    int mc_numRollouts = 500;
    
    //int num_proposal_steps = 10; //doesn't matter for manifold all walk
    double featureWeights[] = {-0.8,  //collision
                                 0,  //tailgate
                               -0.1,  //offroad left
                                 0,  //road left lane
                                 0,  //road center lane
                                 0,  //road right lane
                               -0.1, //offroad right
                                0.0, //car to left of me  //TODO makes things weird!
                                0.0};//car to right of me
                                
//    double featureWeights[] = {-0.254321,0.116144,-0.373926,0.0658465,0.0313395,-0.0467568,-0.0487914,0.0508007,-0.0120739};

//    double featureWeights[] = {  -0.5,  //offroad left
//                                 0,  //road left lane
//                                  0,  //road center lane
//                                 0,  //road right lane
//                               -0.5}; //offroad right
                            
    //generate demonstrations from learned policy
    //could do it two ways (1) just give argmax actions for all entries in qvals or actually drive and record, I'm going to use the second since it matches better with human driving
    vector<pair<string,unsigned int> > demonstration;
    DrivingWorld world(visualize, featureWeights, numFeatures);
     //give world to Q-learner to learn demonstration policy
    TabularQLearner opt_policy(&world, numActions, gamma);
    opt_policy.trainEpoch(numDemoPolicyQsteps, exploreRate, learningRate);
    opt_policy.displayQvalues();
    vector<State> trajectory;

    //record learned policy as a demonstration
    if(givePolicyDemo)
    {
        //give demo for each state argmax action in Q-values
        set<string> states = opt_policy.getQTableStates();
        for(string state : states)
        {
            unsigned int a = opt_policy.getArgmaxQvalues(state);
            pair<string, unsigned int> sa_demo = make_pair(state, a);
            demonstration.push_back(sa_demo);
        }
    }
    else
    {
        world.setVisuals(false); //also visualize the demonstration
        State initState = world.startNewEpoch();
        trajectory.push_back(initState);
        State state = initState;
        cout << "init state: " << initState.toString() << endl;
        for(int step = 0; step < demo_length; step++)
        {
            
            //cout << "============" << step << endl;
            unsigned int action = opt_policy.getArgmaxQvalues(state);
            //save state-action pair in demonstration
            pair<string, unsigned int> sa_demo = make_pair(state.toString(), action);
            if(remove_duplicates)
            {
                //check not in demo already
                if (std::find(demonstration.begin(), demonstration.end(), sa_demo) == demonstration.end())
                    demonstration.push_back(sa_demo);
            }
            else
                demonstration.push_back(sa_demo);
            pair<State,double> nextStateReward = world.updateState(action);
            trajectory.push_back(nextStateReward.first);
            State nextState = nextStateReward.first;
            //double reward = nextStateReward.second;
            state = nextState;
        }
    }

    //print out demonstrations
    cout << "DEMONSTRATIONS" << endl;
    for(pair<string,unsigned int> p : demonstration)
    {
        cout << p.first << ", " << p.second << endl;
    }
    //cout << "Trajectory" << endl;
    //for(State s : trajectory)
    //{
    //    cout << s.toString() << endl;
    //}
    double wfcb = calculateWorstCaseFeatureCountBound(trajectory, &opt_policy, &world, mc_numRollouts, mc_rolloutLength, gamma);
    cout << "WFCB = " << wfcb << endl;
    //qbot.displayQvalues();

//    //intentional segfault
//    State* s;
//    cout << s->toString() << endl;
      
    cout << "RUNNING BIRL" << endl;
    //turn off visualization for birl
    world.setVisuals(false);
    
    double expVal = evaluateExpectedReturn(&opt_policy, &world, mc_numRollouts, mc_rolloutLength, gamma);
    cout << "expected value is " << expVal << endl;
 
//    FeatureBIRL_Q birl(&world, numBIRL_QSteps, exploreRate, learningRate, gamma, min_reward, max_reward, chain_len, mcmc_step, alpha_conf, sample_flag, mcmc_reject, num_proposal_steps);
//    //add demos
//    birl.addPositiveDemos(demonstration);
//    //run birl
//    birl.run();
//    cout << birl.getMAPposterior() << endl;
//    double* mapWeights = birl.getMAPWeights();
//    cout << "MAP weights" << endl;
//    for(int i=0;i<numFeatures; i++)
//        cout << mapWeights[i] << "\t";
//    cout << endl;
// 

//    //view MAP policy learned from BIRL
//    DrivingWorld map_world(visualize, mapWeights, numFeatures);
//    //give world to Q-learner
//    TabularQLearner map_qbot(&map_world, numActions, gamma);
//    map_qbot.trainEpoch(numBIRL_QSteps, exploreRate, learningRate);
//    //test out learned policy
//    map_world.setVisuals(true);
//    State initState = map_world.startNewEpoch();
//    State state = initState;
//    cout << "init state: " << initState.toString() << endl;
//    for(int step = 0; step < 200; step++)
//    {
//        //cout << "============" << step << endl;
//        int action = map_qbot.getArgmaxQvalues(state);
//        pair<State,double> nextStateReward = map_world.updateState(action);
//        State nextState = nextStateReward.first;
//        //double reward = nextStateReward.second;
//        state = nextState;
//    }
//    map_qbot.displayQvalues();
//    //map_world.clean_up();
//    

    world.clean_up();
    return 0;

}
