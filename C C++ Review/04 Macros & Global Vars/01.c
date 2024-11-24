#include <stdio.h>

// examples for each conditional macro
// #if
// #ifdef -- if defined
// #ifndef -- if not defined
// #elif -- else if
// #else -- else
// #endif -- end if condition

#define PI 3.14159
// like lambda function
#define AREA(r) (PI * r * r)

// if not defined then define radius and end if
#ifndef radius
#define radius 7
#endif

// if elif else logic
// we can only use integer constants in #if and #elif
#if radius > 10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 7
#endif


int main() {
    printf("Area of circle with radius %d: %f\n", radius, AREA(radius));  // Output: Area of circle with radius 6.900000: 149.571708
}