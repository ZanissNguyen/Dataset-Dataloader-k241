/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   listheader.h
 * Author: ltsach
 *
 * Created on September 7, 2024, 10:51 PM
 */

#ifndef LISTHEADER_H
#define LISTHEADER_H

#include "XArrayList.h"
#include "DLinkedList.h"
#include <type_traits>

template<class T>
using xvector = XArrayList<T>;
template<class T>
using xlist = DLinkedList<T>;

template<typename T>
bool isPointer()
{
    return std::is_pointer<T>::value;
}

#endif /* LISTHEADER_H */

