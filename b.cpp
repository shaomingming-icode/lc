#include <iostream>

int main(int argc, char* argv[]) {
    int* ptr = new int[1];
    ptr[0] = 0;
  delete[] ptr;
  return 0;
}
