#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include "../include/unit_norm_sampling.hpp"

using namespace std;

void test_up_down()
{
   int dim = 5;
    srand (time(NULL));         //sets current time as seed
//    cout << "testing rand unit ball samples" << endl;
//    for(int rep = 0; rep < 2; rep++)
//    {
//        double* samp = sample_unit_L1_norm(dim);
//        double sum = 0;
//        for(int i=0; i<dim; i++)
//        {
//            cout << samp[i] << " ";
//            sum += abs(samp[i]);
//        }
//        cout << endl;
//        cout << "sum = " << sum << endl;
//    }    
    
    cout << "up down walk on l1 unit ball" << endl;
    cout << "checking [0 0 1]" << endl;
    double* init_sample = new double[dim];
    init_sample[0] = 0.0;
    init_sample[1] = 0.0;
    init_sample[2] = 1.0;
    double step = 0.01;
    for(int rep = 0; rep < 10; rep++)
    {
        double* samp = updown_l1_norm_walk(init_sample, dim, step);
        double sum = 0;
        for(int i=0; i<dim; i++)
        {
            cout << samp[i] << " ";
            sum += abs(samp[i]);
        }
        cout << endl;
        cout << "sum = " << sum << endl;
//        init_sample = samp;
    
    }
    cout << "checking postive and negative values" << endl;
    init_sample[0] = -0.2;
    init_sample[1] = 0.4;
    init_sample[2] = -0.4;
    for(int rep = 0; rep < 10; rep++)
    {
        double* samp = updown_l1_norm_walk(init_sample, dim, step);
        samp = updown_l1_norm_walk(samp, dim, step);
        samp = updown_l1_norm_walk(samp, dim, step);
        double sum = 0;
        for(int i=0; i<dim; i++)
        {
            if(isEqual(samp[i],0.0))
                cout << "0.00" << "\t";
            else
                cout << samp[i] << "\t";
            sum += abs(samp[i]);
        }
        cout << endl;
        //cout << "sum = " << sum << endl;
        assert(isEqual(sum, 1.0));
        init_sample = samp;
    
    }
}

void test_manifold_walk()
{
    double val1 = -0.5;
    double val2 = -0.5;
    string dir = "cntclockwise";
    double step = 0.01;
    pair<double, double> p = manifold_l1_step(val1, val2, dir, step);
    cout << p.first << ", " << p.second << endl;
    
    cout << "----testing full step ----" << endl;
    srand(2);
    int dim = 5;
    double* weights = new double[dim];
    fill(weights, weights + dim, 0);
    weights[0] = 1.0;
    for(int i=0; i<9; i++)
    {
        cout << "-----iter " << i << "--------" << endl;
        double* samp = random_manifold_l1_step(weights, dim, step,10);
        delete weights;
        weights = samp;
        double sum = 0;
        for(int i=0; i<dim; i++)
        {
            cout << samp[i] << " ";
            sum += abs(samp[i]);
        }
        cout << endl;
        assert(isEqual(sum,1.0));
    }

}

int main()
{
    //test_up_down();
    test_manifold_walk();
}



