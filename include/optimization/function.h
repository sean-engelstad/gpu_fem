#pragma once
#include <string>
#include <vector>
#include "variable.h"

template <typename T>
class Function
{
public:
private:
    std::string name;
    T value;
    std::vector<T> dv_sens;
    std::vector<T> xpt_sens;
};