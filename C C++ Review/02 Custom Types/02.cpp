// exact same as 02.c
// #include <stdio.h>
#include <iostream>

using namespace std;

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {1.1, 2.5};
    //for c++
    cout << "Size of Point" << sizeof(Point) << endl;
    // for c
    // printf("size of Point: %zu\n", sizeof(Point));  // Output: 8 bytes = 4 bytes (float x) + 4 bytes (float y)
    
}