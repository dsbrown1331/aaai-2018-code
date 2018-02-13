#include <unordered_map>
#include <string>
#include "../include/q_learner_driving.hpp"
#include "../include/driving_world.hpp"


//TODO need to add features for distance to car in all lanes. I think this will allow the car to prefer one lane and to learn to drive off road and side swipe when cars come by. Currently it tries to prefer one lane until a car comes, then it changes lanes, but then it says "why am I not in the prefered lane?" and comes right back and collides.

//script to test out q-learner
int main()
{
    srand(time(NULL));
    //create world
    bool visualize = false;
    int numStateFeatures = 12;
    int numRewardFeatures = 6;
    bool twoCars = true;
    double exploreRate = 1.0; //without a goal, I think qlearning with epsilon should be close to 1 so we see lots of states and take all possible actions many times
    double learningRate = 0.2;
    int numActions = 3;
    double gamma = 0.9;
    int numSteps = 60000;     
double featureWeights[] = {-0.0766667,	-0.216667,	0.03,	0.396667,	0.116667,	-0.163333};
//    double featureWeights[] = {1,  //collision
//                                 -1,  //tailgate
//                               -1,  //offroad left
//                                 0,  //road left lane
//                                  0,  //road center lane
//                                 0,  //road right lane
//                               -1}; //offroad right
//                            // 0.0, //car to left of me  //TODO makes things weird!
//                            // 0.0};//car to right of me

    DrivingWorld world(visualize, featureWeights, numStateFeatures, numRewardFeatures, twoCars);
    //give world to Q-learner
    TabularQLearner qbot(&world, numActions, gamma);
    qbot.trainEpoch(numSteps, exploreRate, learningRate);
    cout << "done training" << endl;
    //test out learned policy
    world.setVisuals(true);
    State initState = world.startNewEpoch();
    State state = initState;
    cout << "init state: " << initState.toStateString() << endl;
    for(int step = 0; step < 100; step++)
    {
        cout << "============" << step << endl;
        int action = qbot.getArgmaxQvalues(state); //I could use softmax here rather than hard argmax
        pair<State,double> nextStateReward = world.updateState(action);
        State nextState = nextStateReward.first;
        //double reward = nextStateReward.second;
        state = nextState;
    }
    qbot.displayQvalues();
    world.clean_up();
    return 0;

}
