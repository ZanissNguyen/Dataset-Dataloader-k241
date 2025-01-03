/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.h to edit this template
 */

/* 
 * File:   Softmax.h
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:46 PM
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "ann/Layer.h"

class Softmax: public Layer {
public:
    Softmax(int axis=-1);
    Softmax(const Softmax& orig);
    virtual ~Softmax();

    virtual xt::xarray<double> forward(xt::xarray<double> X);
    
private:
    int axis;
    xt::xarray<double> cached_Y;    
};

//Softmax
    // xt::xarray<double> exp_X = xt::exp(X);
    // xt::xarray<double> softMax = exp_X/(xt::sum(exp_X));
    // cout << softMax << endl;
    // cout << xt::sum(softMax) << endl;

#endif /* SOFTMAX_H */

