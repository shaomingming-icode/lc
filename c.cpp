
int main(int argc, char* argv[]) {  // NOLINT
    int* ptr = new int[1];
    ptr[0] = 0;
    delete[] ptr;
    return 0;
}
