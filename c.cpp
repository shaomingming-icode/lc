#include <iostream>
#include <vector>
int main(int argc, char* argv[]) {
  int* ptr = new int[10];
  delete ptr;
  return 0;
}
