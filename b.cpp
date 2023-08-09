#include <iostream>

int main(int argc, char* argv[]) {
  int* ptr = new int[10];
  delete ptr;
  ptr[0] = 0;
  int i = 0;
  if (i == 1) {
  }
  return 0;
}
