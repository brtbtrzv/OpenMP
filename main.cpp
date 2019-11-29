#include <iostream>
#include <assert.h>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <cstdlib>
#include <cstdio>

using namespace std;

void shift_horizontal (vector <vector <int> > &a, int k, int i) {
    unsigned long n = a.size();
    for (int j = 0; j < k / 2; ++j) {
        swap(a[i][(n - k + j)], a[i][(n - j - 1)]);
    }

    for (int j = 0; j < n / 2; ++j) {
        swap(a[i][j],a[i][(n - j - 1)]);
    }

    for (int j = 0; j < (n - k) / 2; ++j) {
        swap(a[i][(k + j)],a[i][(n - j - 1)]);
    }
}

void shift_vertical (vector <vector <int> > &a, int k, int j) {
    unsigned long n = a.size();
    for (int i = 0; i < k / 2; ++i) {
        swap(a[n - k + i][j], a[n - i - 1][j]);
    }

    for (int i = 0; i < n / 2; ++i) {
        swap(a[i][j],a[(n - i - 1)][j]);
    }

    for (int i = 0; i < (n - k) / 2; ++i) {
        swap(a[k + i][j],a[n - i - 1][j]);
    }
}

void matrix_mult (vector <vector <int> > &a, vector <vector<int> > &b, vector <vector<int> > &c,
                  unsigned long idx_i, unsigned long idx_j, int q) {
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < q; ++j) {
            for (int k = 0; k < q; ++k) {
                c[idx_i + i][idx_j + j] += a[idx_i + i][idx_j + k]*b[idx_i + k][idx_j + j];
            }
        }
    }
}

void kannon_alg (vector <vector <int> > &a, vector <vector<int> > &b, vector <vector<int> > &c, unsigned long q) {
    unsigned long n = a.size();
    vector <vector <int> > a1 = a;
    vector <vector <int> > b1 = b;

    unsigned long n1 = n / q;
    int i1, j1;
    for (int i = 0; i < n; ++i) {
        i1 = i / n1;
        shift_horizontal(a, n - n1 * i1, i);
    }
    for (int j = 0; j < n; ++j) {
        j1 = j / n1;
        shift_vertical (b, n - n1 * j1, j);
    }
#pragma omp parallel for private(num_thread, i1, j1) num_threads(q * q)
    unsigned long num_thread = omp_get_thread_num();

    i1 = num_thread / q;
    j1 = num_thread % q;

    for (int iter = 0; iter < q; ++iter) {
        matrix_mult(a, b, c, i1 * n1, j1 * n1, n1);
        for (int i = 0; i < n1; ++i) {
            shift_horizontal(a, n - n1, i1 * n1 + i);
        }
        for (int j = 0; j < n1; ++j) {
            shift_vertical (b, n - n1, j1 * n1 + j);
        }
    }
}

void check_mult(vector <vector <int> > &a, vector <vector<int> > &b, vector <vector<int> > &c) {
    unsigned long n = a.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}



int main() {
    unsigned long sz = 6;
    vector <vector<int> > a(sz, vector<int> (sz));
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            cin >> a[i][j];
        }
    }
//    a = {{1,2,3,4,5,6},
//         {0,1,0,0,0,0},
//         {0,0,1,0,0,0},
//         {0,0,0,1,0,0},
//         {0,0,0,0,1,0},
//         {0,0,0,0,0,1}};

    vector <vector<int> > b(sz, vector<int> (sz));
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            cin >> b[i][j];
        }
    }
//    b = {{1,1,1,1,1,1},
//         {0,1,0,0,0,0},
//         {0,0,1,0,0,0},
//         {0,0,0,1,0,0},
//         {0,0,0,0,1,0},
//         {0,0,0,0,0,1}};

    vector <vector<int> > c(sz, vector<int> (sz));
    unsigned long q = 3;
    double timer = omp_get_wtime();
    kannon_alg(a, b, c, q);
    timer = omp_get_wtime() - timer;
//    for (auto i : c) {
//        for (auto j : i) {
//            cout << j << " ";
//        }
//        cout << endl;
//    }
    vector <vector<int> > c_check(sz, vector<int> (sz));
    check_mult(a, b, c_check);
    if (c == c_check) {
        cout << "OK" << endl;
        cout << timer << "\t\t" << "// OpenMP. nThreads: " << q << " size: " << sz << endl;
        cout << "_________________";
    }
    else {
        cout << "WRONG_ANSWER" << endl;
        cout << "_________________";
    }

    return 0;

}
