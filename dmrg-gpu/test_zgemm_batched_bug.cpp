/**
 * Minimal reproducer: rocblas_zgemm_batched produces wrong results
 * while rocblas_dgemm_batched works correctly with same parameters.
 *
 * Pattern: D iterations of batch_count=d GEMMs with accumulation (beta=0/1).
 * All batch elements share the same B pointer (different A and C pointers).
 *
 * Build: hipcc -o test_zgemm_batched_bug test_zgemm_batched_bug.cpp -lrocblas -I/opt/rocm/include
 * Run:   ./test_zgemm_batched_bug
 */

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if(e) { printf("HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)
#define ROCBLAS_CHECK(x) do { rocblas_status s = (x); if(s) { printf("rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while(0)

// Test parameters matching Josephson: d=3, D=4, small chi
static const int D = 4, d = 3;
static const int cL = 3, cR = 3;  // small chi for easy verification

// Reference CPU computation (complex)
void cpu_step3_complex(
    const hipDoubleComplex* U,   // (D*d) blocks of (cL, cR)
    const hipDoubleComplex* R,   // D blocks of (cR, cR) with stride cR*D
    hipDoubleComplex* result,    // d blocks of (cL, cR) with stride cL*d
    int cL, int cR, int D, int d)
{
    // result_s'[a',b'] = sum_w' U_{w'*d+s'}[a',b] * R_w'[b,b']
    for (int sp = 0; sp < d; sp++) {
        for (int ap = 0; ap < cL; ap++) {
            for (int bp = 0; bp < cR; bp++) {
                hipDoubleComplex acc = make_hipDoubleComplex(0,0);
                for (int wp = 0; wp < D; wp++) {
                    for (int b = 0; b < cR; b++) {
                        // U[ws_out][ap, b] in col-major with lda=cL
                        int ws_out = wp * d + sp;
                        hipDoubleComplex u_val = U[ws_out * cL * cR + b * cL + ap];
                        // R[wp][b, bp] in col-major with ldb=cR*D
                        hipDoubleComplex r_val = R[wp * cR + bp * cR * D + b];
                        acc = hipCadd(acc, hipCmul(u_val, r_val));
                    }
                }
                // result[sp][ap, bp] in col-major with ldc=cL*d
                result[sp * cL + bp * cL * d + ap] = acc;
            }
        }
    }
}

// Reference CPU computation (real)
void cpu_step3_real(
    const double* U, const double* R, double* result,
    int cL, int cR, int D, int d)
{
    for (int sp = 0; sp < d; sp++) {
        for (int ap = 0; ap < cL; ap++) {
            for (int bp = 0; bp < cR; bp++) {
                double acc = 0;
                for (int wp = 0; wp < D; wp++) {
                    for (int b = 0; b < cR; b++) {
                        int ws_out = wp * d + sp;
                        double u_val = U[ws_out * cL * cR + b * cL + ap];
                        double r_val = R[wp * cR + bp * cR * D + b];
                        acc += u_val * r_val;
                    }
                }
                result[sp * cL + bp * cL * d + ap] = acc;
            }
        }
    }
}

double max_diff_complex(const hipDoubleComplex* a, const hipDoubleComplex* b, int n) {
    double mx = 0;
    for (int i = 0; i < n; i++) {
        double dr = fabs(hipCreal(a[i]) - hipCreal(b[i]));
        double di = fabs(hipCimag(a[i]) - hipCimag(b[i]));
        mx = fmax(mx, fmax(dr, di));
    }
    return mx;
}

double max_diff_real(const double* a, const double* b, int n) {
    double mx = 0;
    for (int i = 0; i < n; i++) mx = fmax(mx, fabs(a[i] - b[i]));
    return mx;
}

int main() {
    printf("=== rocblas_zgemm_batched vs rocblas_dgemm_batched bug reproducer ===\n");
    printf("Parameters: D=%d, d=%d, cL=%d, cR=%d\n\n", D, d, cL, cR);

    // Print rocBLAS version
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    srand(42);

    int U_size = D * d * cL * cR;
    int R_size = cR * D * cR;
    int result_size = d * cL * cR;  // with stride cL*d between columns... total = cL*d * cR

    // Actual result buffer: cL*d rows, cR columns
    int result_buf_size = cL * d * cR;

    // ========== COMPLEX TEST ==========
    printf("--- Complex (zgemm_batched) ---\n");
    {
        std::vector<hipDoubleComplex> h_U(U_size), h_R(R_size);
        std::vector<hipDoubleComplex> h_result_cpu(result_buf_size);
        std::vector<hipDoubleComplex> h_result_seq(result_buf_size);
        std::vector<hipDoubleComplex> h_result_bat(result_buf_size);

        for (auto& v : h_U) v = make_hipDoubleComplex((double)rand()/RAND_MAX - 0.5, (double)rand()/RAND_MAX - 0.5);
        for (auto& v : h_R) v = make_hipDoubleComplex((double)rand()/RAND_MAX - 0.5, (double)rand()/RAND_MAX - 0.5);

        // CPU reference
        cpu_step3_complex(h_U.data(), h_R.data(), h_result_cpu.data(), cL, cR, D, d);

        // GPU data
        hipDoubleComplex *d_U, *d_R, *d_result;
        HIP_CHECK(hipMalloc(&d_U, U_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_R, R_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_result, result_buf_size * sizeof(hipDoubleComplex)));

        HIP_CHECK(hipMemcpy(d_U, h_U.data(), U_size * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_R, h_R.data(), R_size * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));

        rocblas_double_complex one = {1.0, 0.0}, zero = {0.0, 0.0};

        // --- Sequential individual zgemm ---
        HIP_CHECK(hipMemset(d_result, 0, result_buf_size * sizeof(hipDoubleComplex)));
        for (int sp = 0; sp < d; sp++) {
            for (int wp = 0; wp < D; wp++) {
                rocblas_double_complex beta = (wp == 0) ? zero : one;
                int ws_out = wp * d + sp;
                ROCBLAS_CHECK(rocblas_zgemm(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR,
                    &one,
                    (const rocblas_double_complex*)(d_U + ws_out * cL * cR), cL,
                    (const rocblas_double_complex*)(d_R + wp * cR), cR * D,
                    &beta,
                    (rocblas_double_complex*)(d_result + sp * cL), cL * d));
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_result_seq.data(), d_result, result_buf_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost));

        // --- Batched zgemm_batched ---
        HIP_CHECK(hipMemset(d_result, 0, result_buf_size * sizeof(hipDoubleComplex)));

        hipDoubleComplex** d_A_ptrs;
        hipDoubleComplex** d_B_ptrs;
        hipDoubleComplex** d_C_ptrs;
        HIP_CHECK(hipMalloc(&d_A_ptrs, d * sizeof(hipDoubleComplex*)));
        HIP_CHECK(hipMalloc(&d_B_ptrs, d * sizeof(hipDoubleComplex*)));
        HIP_CHECK(hipMalloc(&d_C_ptrs, d * sizeof(hipDoubleComplex*)));

        std::vector<hipDoubleComplex*> h_A(d), h_B(d), h_C(d);

        for (int wp = 0; wp < D; wp++) {
            rocblas_double_complex beta = (wp == 0) ? zero : one;
            for (int sp = 0; sp < d; sp++) {
                int ws_out = wp * d + sp;
                h_A[sp] = d_U + ws_out * cL * cR;
                h_B[sp] = d_R + wp * cR;
                h_C[sp] = d_result + sp * cL;
            }
            HIP_CHECK(hipMemcpyAsync(d_A_ptrs, h_A.data(), d*sizeof(hipDoubleComplex*), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_B_ptrs, h_B.data(), d*sizeof(hipDoubleComplex*), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_C_ptrs, h_C.data(), d*sizeof(hipDoubleComplex*), hipMemcpyHostToDevice, stream));

            ROCBLAS_CHECK(rocblas_zgemm_batched(handle,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const rocblas_double_complex**)d_A_ptrs, cL,
                (const rocblas_double_complex**)d_B_ptrs, cR * D,
                &beta,
                (rocblas_double_complex**)d_C_ptrs, cL * d,
                d));
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_result_bat.data(), d_result, result_buf_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost));

        double err_seq = max_diff_complex(h_result_cpu.data(), h_result_seq.data(), result_buf_size);
        double err_bat = max_diff_complex(h_result_cpu.data(), h_result_bat.data(), result_buf_size);
        double err_bat_seq = max_diff_complex(h_result_seq.data(), h_result_bat.data(), result_buf_size);

        printf("  Sequential zgemm vs CPU:  max error = %.3e %s\n", err_seq, err_seq < 1e-12 ? "PASS" : "FAIL");
        printf("  Batched zgemm vs CPU:     max error = %.3e %s\n", err_bat, err_bat < 1e-12 ? "PASS" : "FAIL");
        printf("  Batched vs Sequential:    max error = %.3e %s\n", err_bat_seq, err_bat_seq < 1e-12 ? "MATCH" : "MISMATCH");

        if (err_bat_seq > 1e-12) {
            printf("\n  First mismatching elements:\n");
            int shown = 0;
            for (int i = 0; i < result_buf_size && shown < 5; i++) {
                double dr = fabs(hipCreal(h_result_seq[i]) - hipCreal(h_result_bat[i]));
                double di = fabs(hipCimag(h_result_seq[i]) - hipCimag(h_result_bat[i]));
                if (dr > 1e-12 || di > 1e-12) {
                    printf("    [%d] seq=(%g,%g) bat=(%g,%g) diff=(%g,%g)\n", i,
                        hipCreal(h_result_seq[i]), hipCimag(h_result_seq[i]),
                        hipCreal(h_result_bat[i]), hipCimag(h_result_bat[i]), dr, di);
                    shown++;
                }
            }
        }

        HIP_CHECK(hipFree(d_U)); HIP_CHECK(hipFree(d_R)); HIP_CHECK(hipFree(d_result));
        HIP_CHECK(hipFree(d_A_ptrs)); HIP_CHECK(hipFree(d_B_ptrs)); HIP_CHECK(hipFree(d_C_ptrs));
    }

    // ========== REAL TEST ==========
    printf("\n--- Real (dgemm_batched) ---\n");
    {
        std::vector<double> h_U(U_size), h_R(R_size);
        std::vector<double> h_result_cpu(cL * d * cR);
        std::vector<double> h_result_seq(cL * d * cR);
        std::vector<double> h_result_bat(cL * d * cR);

        for (auto& v : h_U) v = (double)rand()/RAND_MAX - 0.5;
        for (auto& v : h_R) v = (double)rand()/RAND_MAX - 0.5;

        cpu_step3_real(h_U.data(), h_R.data(), h_result_cpu.data(), cL, cR, D, d);

        double *d_U, *d_R, *d_result;
        int buf_size = cL * d * cR;
        HIP_CHECK(hipMalloc(&d_U, U_size * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_R, R_size * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_result, buf_size * sizeof(double)));

        HIP_CHECK(hipMemcpy(d_U, h_U.data(), U_size * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_R, h_R.data(), R_size * sizeof(double), hipMemcpyHostToDevice));

        double one = 1.0, zero_val = 0.0;

        // Sequential
        HIP_CHECK(hipMemset(d_result, 0, buf_size * sizeof(double)));
        for (int sp = 0; sp < d; sp++) {
            for (int wp = 0; wp < D; wp++) {
                double beta = (wp == 0) ? 0.0 : 1.0;
                int ws_out = wp * d + sp;
                ROCBLAS_CHECK(rocblas_dgemm(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR, &one,
                    d_U + ws_out * cL * cR, cL,
                    d_R + wp * cR, cR * D,
                    &beta, d_result + sp * cL, cL * d));
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_result_seq.data(), d_result, buf_size * sizeof(double), hipMemcpyDeviceToHost));

        // Batched
        HIP_CHECK(hipMemset(d_result, 0, buf_size * sizeof(double)));
        double** d_A_ptrs;
        double** d_B_ptrs;
        double** d_C_ptrs;
        HIP_CHECK(hipMalloc(&d_A_ptrs, d * sizeof(double*)));
        HIP_CHECK(hipMalloc(&d_B_ptrs, d * sizeof(double*)));
        HIP_CHECK(hipMalloc(&d_C_ptrs, d * sizeof(double*)));

        std::vector<double*> h_A(d), h_B(d), h_C(d);
        for (int wp = 0; wp < D; wp++) {
            double beta = (wp == 0) ? 0.0 : 1.0;
            for (int sp = 0; sp < d; sp++) {
                int ws_out = wp * d + sp;
                h_A[sp] = d_U + ws_out * cL * cR;
                h_B[sp] = d_R + wp * cR;
                h_C[sp] = d_result + sp * cL;
            }
            HIP_CHECK(hipMemcpyAsync(d_A_ptrs, h_A.data(), d*sizeof(double*), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_B_ptrs, h_B.data(), d*sizeof(double*), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_C_ptrs, h_C.data(), d*sizeof(double*), hipMemcpyHostToDevice, stream));

            ROCBLAS_CHECK(rocblas_dgemm_batched(handle,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                (const double**)d_A_ptrs, cL,
                (const double**)d_B_ptrs, cR * D,
                &beta, d_C_ptrs, cL * d,
                d));
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_result_bat.data(), d_result, buf_size * sizeof(double), hipMemcpyDeviceToHost));

        double err_seq = max_diff_real(h_result_cpu.data(), h_result_seq.data(), buf_size);
        double err_bat = max_diff_real(h_result_cpu.data(), h_result_bat.data(), buf_size);
        double err_bat_seq = max_diff_real(h_result_seq.data(), h_result_bat.data(), buf_size);

        printf("  Sequential dgemm vs CPU:  max error = %.3e %s\n", err_seq, err_seq < 1e-12 ? "PASS" : "FAIL");
        printf("  Batched dgemm vs CPU:     max error = %.3e %s\n", err_bat, err_bat < 1e-12 ? "PASS" : "FAIL");
        printf("  Batched vs Sequential:    max error = %.3e %s\n", err_bat_seq, err_bat_seq < 1e-12 ? "MATCH" : "MISMATCH");

        HIP_CHECK(hipFree(d_U)); HIP_CHECK(hipFree(d_R)); HIP_CHECK(hipFree(d_result));
        HIP_CHECK(hipFree(d_A_ptrs)); HIP_CHECK(hipFree(d_B_ptrs)); HIP_CHECK(hipFree(d_C_ptrs));
    }

    // ========== Variant: non-overlapping C pointers ==========
    printf("\n--- Complex (zgemm_batched) with non-overlapping C layout ---\n");
    {
        std::vector<hipDoubleComplex> h_U(U_size), h_R(R_size);

        srand(42);  // same seed as above
        for (auto& v : h_U) v = make_hipDoubleComplex((double)rand()/RAND_MAX - 0.5, (double)rand()/RAND_MAX - 0.5);
        for (auto& v : h_R) v = make_hipDoubleComplex((double)rand()/RAND_MAX - 0.5, (double)rand()/RAND_MAX - 0.5);

        // Separate result buffers per sp (no interleaving)
        int sep_size = d * cL * cR;
        std::vector<hipDoubleComplex> h_result_bat(sep_size);
        std::vector<hipDoubleComplex> h_result_seq(sep_size);

        hipDoubleComplex *d_U, *d_R, *d_result_sep, *d_result_seq;
        HIP_CHECK(hipMalloc(&d_U, U_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_R, R_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_result_sep, sep_size * sizeof(hipDoubleComplex)));
        HIP_CHECK(hipMalloc(&d_result_seq, sep_size * sizeof(hipDoubleComplex)));

        HIP_CHECK(hipMemcpy(d_U, h_U.data(), U_size * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_R, h_R.data(), R_size * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));

        rocblas_double_complex one = {1.0, 0.0}, zero = {0.0, 0.0};

        // Sequential with separate buffers (ldc=cL, each sp at sp*cL*cR)
        HIP_CHECK(hipMemset(d_result_seq, 0, sep_size * sizeof(hipDoubleComplex)));
        for (int sp = 0; sp < d; sp++) {
            for (int wp = 0; wp < D; wp++) {
                rocblas_double_complex beta = (wp == 0) ? zero : one;
                int ws_out = wp * d + sp;
                ROCBLAS_CHECK(rocblas_zgemm(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR, &one,
                    (const rocblas_double_complex*)(d_U + ws_out * cL * cR), cL,
                    (const rocblas_double_complex*)(d_R + wp * cR), cR * D,
                    &beta,
                    (rocblas_double_complex*)(d_result_seq + sp * cL * cR), cL));
            }
        }

        // Batched with separate buffers (ldc=cL)
        HIP_CHECK(hipMemset(d_result_sep, 0, sep_size * sizeof(hipDoubleComplex)));

        hipDoubleComplex** d_A_ptrs;
        hipDoubleComplex** d_B_ptrs;
        hipDoubleComplex** d_C_ptrs;
        HIP_CHECK(hipMalloc(&d_A_ptrs, d * sizeof(hipDoubleComplex*)));
        HIP_CHECK(hipMalloc(&d_B_ptrs, d * sizeof(hipDoubleComplex*)));
        HIP_CHECK(hipMalloc(&d_C_ptrs, d * sizeof(hipDoubleComplex*)));

        std::vector<hipDoubleComplex*> h_A(d), h_B(d), h_C(d);
        for (int wp = 0; wp < D; wp++) {
            rocblas_double_complex beta = (wp == 0) ? zero : one;
            for (int sp = 0; sp < d; sp++) {
                int ws_out = wp * d + sp;
                h_A[sp] = d_U + ws_out * cL * cR;
                h_B[sp] = d_R + wp * cR;
                h_C[sp] = d_result_sep + sp * cL * cR;  // non-overlapping!
            }
            HIP_CHECK(hipMemcpyAsync(d_A_ptrs, h_A.data(), d*sizeof(hipDoubleComplex*), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_B_ptrs, h_B.data(), d*sizeof(hipDoubleComplex*), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(d_C_ptrs, h_C.data(), d*sizeof(hipDoubleComplex*), hipMemcpyHostToDevice, stream));

            ROCBLAS_CHECK(rocblas_zgemm_batched(handle,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                (const rocblas_double_complex**)d_A_ptrs, cL,
                (const rocblas_double_complex**)d_B_ptrs, cR * D,
                &beta,
                (rocblas_double_complex**)d_C_ptrs, cL,
                d));
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_result_seq.data(), d_result_seq, sep_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_result_bat.data(), d_result_sep, sep_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost));

        double err = max_diff_complex(h_result_seq.data(), h_result_bat.data(), sep_size);
        printf("  Batched vs Sequential (non-overlapping C): max error = %.3e %s\n", err, err < 1e-12 ? "MATCH" : "MISMATCH");
        printf("  (If MATCH: bug is specific to overlapping/strided C pointers)\n");

        HIP_CHECK(hipFree(d_U)); HIP_CHECK(hipFree(d_R));
        HIP_CHECK(hipFree(d_result_sep)); HIP_CHECK(hipFree(d_result_seq));
        HIP_CHECK(hipFree(d_A_ptrs)); HIP_CHECK(hipFree(d_B_ptrs)); HIP_CHECK(hipFree(d_C_ptrs));
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));

    printf("\nDone.\n");
    return 0;
}
