#include <unordered_map>
#include <string>
#include "../include/q_learner_driving.hpp"
#include "../include/driving_world.hpp"
#include "../include/feature_birl_qlearning.hpp"
#include "../include/confidence_bounds_qlearning.hpp"
#include <algorithm>
#include <fstream>

//TODO delete featureWeights at end?
//TODO delete World too


//carExperiment1_1 has chain_len 1000 reps = 50 demo_length = 200 alpha = 10
//carExperiment1_2 has alpha = 100
//carExperiment1_3 has alpha = 100 and chain_len = 5000

//DONE need to add features for distance to car in all lanes. I think this will allow the car to prefer one lane and to learn to drive off road and side swipe when cars come by. Currently it tries to prefer one lane until a car comes, then it changes lanes, but then it says "why am I not in the prefered lane?" and comes right back and collides.
//TODO probably need more discrete distances to car so it knows where it is, otherwise it collides, but doesn't know where it hit 


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
    bool twoCars = true;
    double exploreRate = 0.4; //without a goal, I think qlearning with epsilon should be close to 1 so we see lots of states and take all possible actions many times
    double learningRate = 0.3;
    int numActions = 3;
    double gamma = 0.9;
    int numQSteps = 6000; 
    int demo_length = 100;
    double min_reward = -2;
    double max_reward = 2;
    unsigned int chain_len = 2000; 
    double mcmc_step = 0.01;
    double alpha_conf = 10;
    int sample_flag = 4;
    bool mcmc_reject = true;
    bool remove_duplicates = true;
    //bool givePolicyDemo = false;

    int num_proposal_steps = 10; //doesn't matter for manifold all walk



    double featureWeights[] = {-0.4,  //collision
                                 -0.2,  //tailgate
                               -0.2,  //offroad left
                                 0,  //road left lane
                                  0,  //road center lane
                                 0,  //road right lane
                               -0.2}; //offroad right
                            // 0.0, //car to left of me  //TODO makes things weird!
                            // 0.0};//car to right of me
                  
        //generate demonstrations from learned policy
        //could do it two ways (1) just give argmax actions for all entries in qvals or actually drive and record, I'm going to use the second since it matches better with human driving
    vector<pair<string,unsigned int> > demonstration;
    DrivingWorld world(visualize, featureWeights, numStateFeatures, numRewardFeatures, twoCars);
    DrivingWorld birl_world(visualize, featureWeights, numStateFeatures, numRewardFeatures, twoCars); //use this world for BIRL iterations since it changes rewards at each step
    vector<State> trajectory;
    cout << "Initialized world" << endl;
 
  ////generate expert demonstration
    cout << "Training expert policy" << endl;
    TabularQLearner opt_policy(&world, numActions, gamma);
    opt_policy.trainEpoch(numQSteps, exploreRate, learningRate);
    demonstration.clear();  
    trajectory.clear();
    cout << "Generating demonstration" << endl;
    world.setVisuals(false);
    State initState = world.startNewEpoch();
    trajectory.push_back(initState);
    State state = initState;
    //cout << "init state: " << initState.toString() << endl;
    for(int step = 0; step < demo_length; step++)
    {
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
        pair<State,double> nextStateReward = world.updateState(action);
        trajectory.push_back(nextStateReward.first);
        State nextState = nextStateReward.first;
        //double reward = nextStateReward.second;
        state = nextState;
    }
    //print out demonstrations
    cout << "DEMONSTRATIONS" << endl;
    for(pair<string,unsigned int> p : demonstration)
    {
        cout << p.first << ", " << p.second << endl;
    }
    //qbot.displayQvalues();

    for(unsigned int rep = 0; rep < reps; rep++)
    {
        //set up file for output
        string filename = "alpha" + to_string((int)alpha_conf) 
                        + "_chain" + to_string(chain_len) 
                        + "_step" + to_string(mcmc_step)
                        + "_L1sampleflag" + to_string(sample_flag) 
                        + "_demoLength" + to_string(demo_length)
                        + "_mcRollout" + to_string(mc_numRollouts)
                        + "_rep" + to_string(rep)+ ".txt";
        cout << filename << endl; 
        ofstream outfile("data/carExperiment2_1/" + filename);
    
        srand(startSeed + 3*rep);
        cout << "------Rep: " << rep << "------" << endl;

         
        

      
       


        cout << "RUNNING BIRL" << endl;
        //turn off visualization for birl
        birl_world.setVisuals(false);
     
        FeatureBIRL_Q birl(&birl_world, numQSteps, exploreRate, learningRate, gamma, min_reward, max_reward, chain_len, mcmc_step, alpha_conf, sample_flag, mcmc_reject, num_proposal_steps);
        //add demos
        birl.addPositiveDemos(demonstration);
        //run birl
        birl.run();
        cout << birl.getMAPposterior() << endl;
        double* mapWeights = birl.getMAPWeights();
        cout << "MAP weights" << endl;
        for(int i=0;i<numRewardFeatures; i++)
            cout << mapWeights[i] << "\t";
        cout << endl;
     

        //view MAP policy learned from BIRL
        DrivingWorld map_world(visualize, mapWeights, numStateFeatures, numRewardFeatures, twoCars);
        //give world to Q-learner
        TabularQLearner map_policy(&map_world, numActions, gamma);
        map_policy.trainEpoch(numQSteps, exploreRate, learningRate);
        cout << "learned map policy" << endl;
        //test out learned policy
        if(debug)
        {
            map_world.setVisuals(true);
            State initState = map_world.startNewEpoch();
            State state = initState;
            cout << "init state: " << initState.toStateString() << endl;
            for(int step = 0; step < 100; step++)
            {
                //cout << "============" << step << endl;
                int action = map_policy.getArgmaxQvalues(state);
                pair<State,double> nextStateReward = map_world.updateState(action);
                State nextState = nextStateReward.first;
                //double reward = nextStateReward.second;
                state = nextState;
            }
            //map_policy.displayQvalues();
            //map_world.clean_up();
            map_world.setVisuals(false);
        }
        cout << "writing data to file" << endl;



         ///compute actual expected return difference
        //double opt_V = evaluateExpectedReturn(&opt_policy, &world, mc_numRollouts, mc_rolloutLength, gamma);
        //make this just a Q-table look up
        //first find initial state
        State startState = world.startNewEpoch();
        //Then get opt_policy action in this state
        unsigned int start_action = opt_policy.getArgmaxQvalues(startState);
        //then look up value in Q-table
        double opt_V = opt_policy.getQValue(startState,start_action);
        
        cout << "computed opt_V = " << opt_V << endl;
        double eval_V = evaluateExpectedReturn(&map_policy, &world, mc_numRollouts, mc_rolloutLength, gamma);
        cout << "computed eval_V = " << eval_V << endl;
        double trueDiff = abs(opt_V - eval_V);
        cout << "True difference: " << trueDiff << endl;
        outfile << "#true value --- wfcb --- mcmc ratios" << endl;
        outfile << trueDiff << endl;
        outfile << "---" << endl;
        //compute worst-case feature count bound
        double wfcb = calculateWorstCaseFeatureCountBound(trajectory, &map_policy, &world, mc_numRollouts, mc_rolloutLength, gamma);
        cout << "WFCB: " << wfcb << endl;
        outfile << wfcb << endl;
        outfile << "---" << endl;


        //Calculate differences and output them to file in format true\n---\ndata
        for(unsigned int i=0; i<chain_len; i++)
        {
            if(i%50 == 0)
                cout << i << endl;
            //cout.precision(5);
            //get sampleMDP from chain
            double* sampleFeatureWeights = (*(birl.getRewardChain() + i));
//            cout << "sample weights" << endl;
//            for(int i=0;i<numRewardFeatures; i++)
//                cout << sampleFeatureWeights[i] << "\t";
//            cout << endl;
            //learn optimal policy for sampled reward
            DrivingWorld sample_world(visualize, sampleFeatureWeights, numStateFeatures, numRewardFeatures, twoCars); //the evalFeatureWeights get overwritten
            TabularQLearner sample_policy(&sample_world, numActions, gamma);
            sample_policy.trainEpoch(numQSteps, exploreRate, learningRate);
            //double Vstar = evaluateExpectedReturn(&sample_policy, &sample_world, mc_numRollouts, mc_rolloutLength, gamma);
            //query Q-values
            State startState = sample_world.startNewEpoch();
            //Then get sample_policy action in this state
            unsigned int start_action = sample_policy.getArgmaxQvalues(startState);
            //then look up value in Q-table
            double Vstar = sample_policy.getQValue(startState, start_action);

            //cout << "True Exp Val" << endl;
            //cout << Vstar << endl;
            //cout << "Eval Policy" << endl; 
            double Vhat = evaluateExpectedReturn(&map_policy, &sample_world, mc_numRollouts, mc_rolloutLength, gamma);
            //TODO: why did i calculate it again?
            //double EVD = calculateExpectedValueDifference(&sample_policy, &map_policy, &sample_world, mc_numRollouts, mc_rolloutLength, gamma); 
            //cout << Vhat << endl;
            double VabsDiff = abs(Vstar - Vhat);
//            cout << "abs diff: " << VabsDiff << endl;
            //cout << "evd feature diff" << EVD << endl;
            outfile << VabsDiff << endl;
        }    
        
    }
    world.clean_up();
    return 0;

}
