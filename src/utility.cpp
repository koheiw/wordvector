#include <thread>

// [[Rcpp::export]]
int cpp_get_max_thread() {
    int n = std::thread::hardware_concurrency();
    return n;
}