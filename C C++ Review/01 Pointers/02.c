#include <stdio.h>

int main() {
    int value = 42;
    int* ptr1 = &value;
    int** ptr2 = &ptr1;         // pointer to a pointer
    int*** ptr3 = &ptr2;        // pointer to a pointer to a pointer
    
    
    printf("Value: %d\n", ***ptr3);  // Output: 42
}