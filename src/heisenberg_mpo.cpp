#include <vector>
#include <complex>
#include <cmath>

using namespace std;

// Build Heisenberg spin-1/2 MPO  
vector<vector<vector<vector<complex<double>>>>> build_heisenberg_mpo(int L) {
    // MPO tensor dimensions: [left_bond, physical, physical, right_bond]
    // Bond dimension = 3 for Heisenberg
    const int d = 2;  // physical dimension (spin-1/2)
    const int D_mpo = 3;  // MPO bond dimension
    
    vector<vector<vector<vector<complex<double>>>>> mpo(L);
    
    // Pauli matrices
    vector<vector<complex<double>>> sx = {{0, 1}, {1, 0}};
    vector<vector<complex<double>>> sy = {{0, complex<double>(0,-1)}, {complex<double>(0,1), 0}};
    vector<vector<complex<double>>> sz = {{1, 0}, {0, -1}};
    vector<vector<complex<double>>> eye = {{1, 0}, {0, 1}};
    
    for (int site = 0; site < L; site++) {
        mpo[site].resize(D_mpo);
        for (int a = 0; a < D_mpo; a++) {
            mpo[site][a].resize(d);
            for (int i = 0; i < d; i++) {
                mpo[site][a][i].resize(d);
                for (int j = 0; j < d; j++) {
                    mpo[site][a][i][j].resize(D_mpo, 0.0);
                }
            }
        }
        
        // Fill MPO tensors (Heisenberg Hamiltonian)
        if (site == 0) {
            // Left boundary
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][1] = 0.5 * sx[i][j];
                    mpo[site][0][i][j][2] = 0.5 * sy[i][j];
                }
            }
        } else if (site == L-1) {
            // Right boundary  
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][1][i][j][0] = sx[i][j];
                    mpo[site][2][i][j][0] = sy[i][j];
                }
            }
        } else {
            // Bulk sites
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = eye[i][j];
                    mpo[site][0][i][j][1] = 0.5 * sx[i][j];
                    mpo[site][0][i][j][2] = 0.5 * sy[i][j];
                    mpo[site][1][i][j][0] = sx[i][j];
                    mpo[site][2][i][j][0] = sy[i][j];
                }
            }
        }
    }
    
    return mpo;
}
