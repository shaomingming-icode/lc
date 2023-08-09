#include <iostream>

int main(int argc, char* argv[]) {
  int* ptr = new int[10];
  delete ptr;
  ptr[0] = 0;
  return 0;
}
