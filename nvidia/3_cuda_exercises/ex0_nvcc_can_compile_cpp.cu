/// @brief on the purpose of this exercise:
// 1. to show that nvcc can compile regular cpp files

#include <vector>
#include <iostream>

using namespace std;

int main() {

    vector<int> a;
    a.push_back(20);
    cout << a[0] << endl;
}