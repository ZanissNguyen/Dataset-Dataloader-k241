#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "list/listheader.h"
#include "ann/dataset.h"

#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>

#include "list/XArrayListDemo.h"
#include "list/DLinkedListDemo.h"
#include "ann/dataloader.h"

using namespace std;

int main(int argc, char** argv) {
    cout << "Assignment-1" << endl;

    // DLinkedList<int> list;
    // list.println();
    // list.add(100);
    // list.println();
    // list.add(0, 999);
    // list.println();
    // list.clear();
    // list.println();
    // cout << "[ " <" ]";

    // dlistDemo1();
    // dlistDemo1a();

    // unsigned long j;
    // srand( (unsigned)time(NULL) );

    // for( j = 0; j < 100; ++j )
    // {
    //     int n;

    //     /* skip rand() readings that would make n%6 non-uniformly distributed
    //       (assuming rand() itself is uniformly distributed from 0 to RAND_MAX) */
    //     while( ( n = rand() ) > RAND_MAX - (RAND_MAX-5)%6 )
    //     { /* very unlikely event that bad value retrieved
    //                             so we proceed to a next one */
    //     }

    //     printf( "%d,\t%d\n", n, n % 6 + 1 );
    // }

    // // ImageFolderDataset<double, int> test("test");

    // xt::xarray<int> test = xt::random::randint({90, 5, 5}, 0, 20);

    // cout << "Test Shape " <<xt::adapt(test.shape()) << endl;
    // // cout << test << endl;
    // cout << xt::view(test, 39) << endl;

    // xt::xarray<int> a = xt::view(test, xt::range(20, 40));

    // cout << "A Shape " << xt::adapt(a.shape()) << endl;
    // // cout << a << endl;
    // cout << xt::view(a, 19) << endl;

    // int nsamples = 90;
    // bool drop_last = true;
    // bool shuffle = true;
    // xt::xarray<double> X = xt::random::randn<double>({nsamples, 10});
    // xt::xarray<double> T = xt::random::randn<double>({nsamples, 5});
    // TensorDataset<double, double> ds(X, T);

    // cout << ds.len() << endl;

    // int batch_size = 20;
    // int batch_amount = nsamples/batch_size;
    // int modBatch = nsamples%batch_size;

    // DLinkedList<Batch<double, double>> batches;

    // cout << xt::view(X, 1) << endl;
    // cout << xt::view(T, 1) << endl;

    // if (shuffle)
    // {
    //     srand( (unsigned)time(NULL) );
    //     int seed = rand();
    //     xt::random::seed(seed);
    //     xt::random::shuffle(X);
    //     xt::random::seed(seed);
    //     xt::random::shuffle(T);
    // }

    // cout << xt::view(X, 1) << endl;
    // cout << xt::view(T, 1) << endl;

    // for (int i = 0; i<batch_amount; i++)
    // {
    //     int extra = modBatch * ((i==batch_amount-1 && !drop_last) ? 1 : 0);
    //     cout << extra << endl;

    //     Batch<double, double> toCreate(xt::view(X, xt::range(i*batch_size, (i+1)*batch_size + extra))
    //                                 , xt::view(T, xt::range(i*batch_size, (i+1)*batch_size + extra)));
    //     cout << xt::adapt(toCreate.getData().shape()) << endl;
    
    //     batches.add(toCreate);
    // } 

    // cout << batch_amount << " " << modBatch << endl;

    int nsamples = 100;
    xt::xarray<double> X = xt::random::randn<double>({nsamples, 10});
    xt::xarray<double> T = xt::random::randn<double>({nsamples, 5});
    TensorDataset<double, double> * ds = new TensorDataset<double, double>(X, T);
    DataLoader<double, double> loader(ds, 30, true, false);
    cout << loader.getBatches().size() << endl;
    // for (auto it=loader.begin(); it!=loader.end(); it++)
    // {
    //     cout << xt::adapt((*it).getData().shape()) << endl;
    // }

    for(auto batch: loader)
    {
        cout << shape2str(batch.getData().shape()) << endl;
        cout << shape2str(batch.getLabel().shape()) << endl;
    }

    // ReLU
    // xt::xarray<double> X = xt::random::randn<double>({100, 1});
    // xt::xarray<double> T = xt::zeros_like(X);
    // cout << X << endl;
    // cout << T << endl;
    // cout << xt::maximum(X, T) << endl;

    //Softmax
    // xt::xarray<double> exp_X = xt::exp(X);
    // xt::xarray<double> softMax = exp_X/(xt::sum(exp_X));
    // cout << softMax << endl;
    // cout << xt::sum(softMax) << endl;

    // normal
    // xt::xarray<double> inputs = xt::random::randn<double>({1, 100});
    // xt::xarray<double> weights = xt::random::randn<double>({100, 30});
    // xt::xarray<double> bias = xt::random::randn<double>({1, 30});
    // xt::xarray<double> result =  xt::linalg::dot(inputs, weights) + bias;
    // cout << result << endl;
    // cout << xt::adapt(result.shape()) << endl;

    return 0;
}

