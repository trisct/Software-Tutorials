#include<iostream>
#include<chrono>

using namespace std;

int main() {
    float a = 3215.35127;
    float b = 3.;
    float c;
    
    auto start = chrono::steady_clock::now();
    for(long long i=0;i<1000000000LL; i++) {
        c = a / b;
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> time_elapsed = end - start;
    std::cout << "Time elapsed (div ver): "<< time_elapsed.count() << endl;
    
    a = 3215.35127;
    b = 1./3.;
    start = chrono::steady_clock::now();
    for(long long i=0;i<1000000000LL; i++) {
        c = a * b;
    }
    end = chrono::steady_clock::now();
    time_elapsed = end - start;
    std::cout << "Time elapsed (mul ver): "<< time_elapsed.count() << endl;

}

