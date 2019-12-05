#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <stdio.h>

void shift_horizontal (std::vector <std::vector <int> > &a, int k, int i) {
    unsigned long n = a.size();
    for (int j = 0; j < k / 2; ++j) {
        std::swap(a[i][(n - k + j)], a[i][(n - j - 1)]);
    }

    for (int j = 0; j < n / 2; ++j) {
        std::swap(a[i][j],a[i][(n - j - 1)]);
    }

    for (int j = 0; j < (n - k) / 2; ++j) {
        std::swap(a[i][(k + j)],a[i][(n - j - 1)]);
    }
}

void shift_vertical (std::vector <std::vector <int> > &a, int k, int j) {
    unsigned long n = a.size();
    for (int i = 0; i < k / 2; ++i) {
        std::swap(a[n - k + i][j], a[n - i - 1][j]);
    }

    for (int i = 0; i < n / 2; ++i) {
        std::swap(a[i][j],a[(n - i - 1)][j]);
    }

    for (int i = 0; i < (n - k) / 2; ++i) {
        std::swap(a[k + i][j],a[n - i - 1][j]);
    }
}

void matrix_mult (std::vector <std::vector <int> > &a, std::vector <std::vector<int> > &b, std::vector <std::vector<int> > &c,
                  int idx_i, int idx_j, int q) {
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < q; ++j) {
            for (int k = 0; k < q; ++k) {
                c[idx_i + i][idx_j + j] += a[idx_i + i][idx_j + k]*b[idx_i + k][idx_j + j];
            }
        }
    }
}



void kannon_alg (std::vector <std::vector <int> > a, std::vector <std::vector<int> > b, std::vector <std::vector<int> > &c, int q) {
    int n = a.size();
    int n1 = n / q;
    int i1, j1;
    for (int i = 0; i < n; ++i) {
        i1 = i / n1;
        shift_horizontal(a, n - n1 * i1, i);
    }
    for (int j = 0; j < n; ++j) {
        j1 = j / n1;
        shift_vertical (b, n - n1 * j1, j);
    }
    omp_set_num_threads(q * q);
    for (int iter = 0; iter < q; ++iter) {
        #pragma omp parallel shared (a,b,c)
        {
            int num_thread = omp_get_thread_num();
            int i2 = num_thread / q;
            int j2 = num_thread % q;
            matrix_mult(a, b, c, i2 * n1, j2 * n1, n1);
        };
        for (int i = 0; i < n; ++i) {
            shift_horizontal(a, n - n1, i);
        }
        for (int j = 0; j < n; ++j) {
            shift_vertical(b, n - n1, j);
        }
    }
}

void check_mult(std::vector <std::vector <int> > &a, std::vector <std::vector<int> > &b, std::vector <std::vector<int> > &c_check) {
    unsigned long n = a.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c_check[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}



int main(int argc, char **argv) {
    unsigned long sz = 315;
    int q_arr[] = {1, 3, 5, 7, 9};
    for (int idx = 0; idx < 5; ++idx) {
        int q = q_arr[idx];
        int num_cores = (q_arr[idx] * q_arr[idx]) / 8 + 1;

        std::vector <std::vector<int> > a(sz, std::vector<int> (sz));
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) {
                a[i][j] = rand();
            }
        }


        std::vector <std::vector<int> > b(sz, std::vector<int> (sz));
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) {
                b[i][j] = rand();
            }
        }

        std::vector <std::vector<int> > c(sz, std::vector<int> (sz));
        double timer = omp_get_wtime();
        kannon_alg(a, b, c, q);
        timer = omp_get_wtime() - timer;

        std::vector <std::vector<int> > c_check(sz, std::vector<int> (sz));
//    double timer1 = omp_get_wtime();
//    check_mult(a, b, c_check);
//    timer1 = omp_get_wtime() - timer1;

        std::cout << num_cores << " " << std::setprecision(5) << timer << "\t\t" << "// OpenMP. nThreads: " << q * q << " size: " << sz << std::endl;
//    if (c == c_check) {
//        std::cout << "OK" << std::endl;
//        std::cout << std::setprecision(5) << timer;
//        std :: cout << " vs ";
//        std::cout << std::setprecision(5) << timer1 << "\t\t" << "// OpenMP. nThreads: " << q * q << " size: " << sz << std::endl;
//        std::cout << "_________________" << std::endl;
//    }
//    else {
//        std::cout << std::setprecision(5) << timer << std::endl;
//        std::cout << " matrix C:" << std::endl;
//        for (auto i : c_check) {
//            for (auto j : i) {
//                std::cout << j << " ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << "matrix C_check:" << std::endl;
//        for (auto i : c_check) {
//            for (auto j : i) {
//                std::cout << j << " ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//        std::cout << "WRONG_ANSWER" << std::endl;
//        std::cout << "_________________" << std::endl;
//    }

    }


    return 0;

}
