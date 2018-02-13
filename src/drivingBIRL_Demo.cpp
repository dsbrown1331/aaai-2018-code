#include <unordered_map>
#include <string>
#include "../include/q_learner_driving_demo.hpp"
#include "../include/driving_world_demo.hpp"
#include "../include/feature_birl_qlearning_demo.hpp"
#include <algorithm>





//DONE need to add features for distance to car in all lanes. I think this will allow the car to prefer one lane and to learn to drive off road and side swipe when cars come by. Currently it tries to prefer one lane until a car comes, then it changes lanes, but then it says "why am I not in the prefered lane?" and comes right back and collides.
//TODO probably need more discrete distances to car so it knows where it is, otherwise it collides, but doesn't know where it hit 

//script to test out q-learner in BIRL
int main(int argc, char *argv[])
{
    srand(time(NULL));
    //srand(123);
    //create world
    bool visualize = false;
    int numStateFeatures = 12;
    int numRewardFeatures = 6;
    bool twoCars = true;
    double exploreRate = 0.8; //without a goal, I think qlearning with epsilon should be close to 1 so we see lots of states and take all possible actions many times
    double learningRate = 0.1;
    int numActions = 3;
    double gamma = 0.99;
    int numDemoPolicyQsteps = 6000;
    int numBIRL_QSteps = 6000;     
    int demo_length = 100;
    double min_reward = -2;
    double max_reward = 2;
    unsigned int chain_len = 1200;
    double mcmc_step = 0.01;
    double alpha_conf = 5;
    int sample_flag = 4;
    bool mcmc_reject = true;
    bool remove_duplicates = true;
    bool giveHumanDemo = true;
    bool trainingRound = true;
    int num_proposal_steps = 10; //doesn't matter for manifold all walk
    double featureWeights[] = {-0.5,  //collision
                             //    -0.1,  //tailgate
                               -0.1,  //offroad left
                                 0,  //road left lane
                                  0,  //road center lane
                                 0.01,  //road right lane
                               -0.1}; //offroad right
                            // 0.0, //car to left of me  //TODO makes things weird!
                            // 0.0};//car to right of me
                                
//    double featureWeights[] = {-0.254321,0.116144,-0.373926,0.0658465,0.0313395,-0.0467568,-0.0487914,0.0508007,-0.0120739};

//    double featureWeights[] = {  -0.5,  //offroad left
//                                 0,  //road left lane
//                                  0,  //road center lane
//                                 0,  //road right lane
//                               -0.5}; //offroad right
                            
    //generate demonstrations from learned policy
    //could do it two ways (1) just give argmax actions for all entries in qvals or actually drive and record, I'm going to use the second since it matches better with human driving
    vector<pair<string,unsigned int> > demonstration;
    DrivingWorld world(visualize, featureWeights, numStateFeatures, numRewardFeatures, twoCars);
    DrivingWorld demo_world(visualize, featureWeights, numStateFeatures, numRewardFeatures, twoCars);


    if(argc >=2)
    {   
        cout << "training" << endl;
        DrivingWorld trial(visualize, featureWeights, numStateFeatures, numRewardFeatures, twoCars);
        trial.setSplashScreen("Practice Mode");
        cout << "set up" << endl;
        trial.waitOnKeyPress();
        usleep(100000);
        cout << "done sleep" << endl;
        trial.startHumanDemo(2000, false);
        trial.clean_up();
        usleep(100000);
    }
    cout << "collecting demos" << endl;
    //record learned policy as a demonstration
    if(giveHumanDemo)
    {
        vector<pair<string,unsigned int> > demo;
        
        while(demo.size() == 0)
        {
            demo_world.setSplashScreen("Teaching Mode");
            cout << "set up" << endl;
            demo_world.waitOnKeyPress();
            usleep(100000);
            demo = demo_world.startHumanDemo(2000, true);
            
        }
        cout << "RAW DEMONSTRATION" << endl;
        for(pair<string,unsigned int> p : demo)
        {
            cout << p.first << ", " << p.second << endl;
        }
        cout << "------------" << endl;
        for(pair<string, unsigned int> sa_demo : demo)
        {
            if(remove_duplicates)
            {
                //check not in demo already
                if (std::find(demonstration.begin(), demonstration.end(), sa_demo) == demonstration.end())
                    demonstration.push_back(sa_demo);
            }
            else
                demonstration.push_back(sa_demo);
        }
    }
    else
    {
        //give world to Q-learner to learn demonstration policy
        TabularQLearner qbot(&demo_world, numActions, gamma);
        qbot.trainEpoch(numDemoPolicyQsteps, exploreRate, learningRate);
        //qbot.displayQvalues();

        demo_world.setVisuals(true); //also visualize the demonstration
        State initState = demo_world.startNewEpoch();
        State state = initState;
        //cout << "init state: " << initState.toStateString() << endl;
        for(int step = 0; step < demo_length; step++)
        {
            //cout << "============" << step << endl;
            unsigned int action = qbot.getArgmaxQvalues(state);
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
            pair<State,double> nextStateReward = demo_world.updateState(action);
            State nextState = nextStateReward.first;
            //double reward = nextStateReward.second;
            state = nextState;
        }
    }
    demo_world.clean_up();

    //print out demonstrations
    cout << "DEMONSTRATIONS" << endl;
    for(pair<string,unsigned int> p : demonstration)
    {
        cout << p.first << ", " << p.second << endl;
    }
    //qbot.displayQvalues();

//    //intentional segfault
//    State* s;
//    cout << s->toString() << endl;
      
    cout << "RUNNING BIRL" << endl;
    //turn off visualization for birl
    world.setVisuals(false);
 
    FeatureBIRL_Q birl(&world, numBIRL_QSteps, exploreRate, learningRate, gamma, min_reward, max_reward, chain_len, mcmc_step, alpha_conf, sample_flag, mcmc_reject, num_proposal_steps);
    //add demos
    birl.addPositiveDemos(demonstration);
    //run birl
    birl.run();
    //cout << birl.getMAPposterior() << endl;
    double* mapWeights = birl.getMAPWeights();
    cout << "MAP weights" << endl;
    for(int i=0;i<numRewardFeatures; i++)
        cout << mapWeights[i] << "\t";
    cout << endl;
     
//world.clean_up();

    cout << "optimizing MAP reward" << endl;
    //view MAP policy learned from BIRL
    DrivingWorld map_world(visualize, mapWeights, numStateFeatures, numRewardFeatures, twoCars);
    //give world to Q-learner
    TabularQLearner map_qbot(&map_world, numActions, gamma);
    map_qbot.trainEpoch(numDemoPolicyQsteps, exploreRate, learningRate);
    //test out learned policy
    map_world.setVisuals(true);
    State initState = map_world.startNewEpoch(); 
    //cout << initState.toStateString() << endl;
    State state = initState;
    //cout << "init state: " << initState.toStateString() << endl;
    for(int step = 0; step < 100; step++)
    {
        //cout << "============" << step << endl;
        int action = map_qbot.getArgmaxQvalues(state);
	//cout << action << endl;
        pair<State,double> nextStateReward = map_world.updateState(action);
        State nextState = nextStateReward.first;
	//if(nextState.getRewardFeature(0) == 1)
	//	cout << "Collision" << endl;
	//cout << nextState.toStateString() << endl;
        //double reward = nextStateReward.second;
        state = nextState;
    }
    //map_qbot.displayQvalues();
    //map_world.clean_up();
    


    return 0;

}
