#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void permuteRandomly(int* values, int numValues);
vector<pair<int,int> > generateAllUniquePairs(int* values, int numValues);



int main()
{
    srand(time(NULL));    
    
    
    int numFeatures = 5;
    int permutation[numFeatures];
    for(int i=0; i<numFeatures; i++)
        permutation[i] = i;
    permuteRandomly(permutation, numFeatures);   
    for(int i=0; i<numFeatures; i++)
        cout << permutation[i] << " ";
    cout << endl;
    vector<pair<int, int> > tuples = generateAllUniquePairs(permutation, numFeatures);
    for(pair<int,int> t : tuples)
        cout << t.first << ", " << t.second << endl;

    return 0;
}


//randomly shuffle elements
void permuteRandomly(int* values, int numValues)
{
    int unsorted = numValues-1;
    while(unsorted > 0)
    {
        //pick random number in [0, unsorted) 
        int randIndex = rand() % unsorted;
        //send value at random index to unsorted index and decrement unsorted
        int temp = values[randIndex];
        values[randIndex] = values[unsorted];
        values[unsorted] = temp;
        unsorted--;
    }    
    
}


vector<pair<int,int> > generateAllUniquePairs(int* values, int numValues)
{
    //calculate numValues choose 2
    int numPairs = numValues * (numValues - 1) / 2;
    vector<pair<int, int> > pairs;
    for(int i = 0; i < numValues; i++)
        for(int j = i+1; j < numValues; j++)
            pairs.push_back(make_pair(values[i],values[j]));
    return pairs;

}
