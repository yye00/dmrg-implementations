// Canonical GPU performance scaffolding — shared across all GPU DMRG variants.
// DO NOT add variant-specific code here.
#ifndef GPU_OPTS_H
#define GPU_OPTS_H

#include <hip/hip_runtime.h>
#include <vector>
#include <cstdlib>
#include <cstdio>

// ============================================================================
// GpuOpts — env-var-gated performance toggles for ablation studies.
// Each optimization is off by default so the baseline binary reproduces
// the original numerics exactly. Flip a flag via DMRG_GPU_OPT_<NAME>=1.
// ============================================================================
struct GpuOpts {
    bool device_k      = false;  // skip D2H readback of truncation rank (DMRG_GPU_OPT_DEVICE_K)
    bool lanczos_fixed = false;  // fixed Lanczos iter count, no convergence-check sync (DMRG_GPU_OPT_LANCZOS_FIXED)
    bool lanczos_graph = false;  // capture Lanczos inner loop as a HIP graph (DMRG_GPU_OPT_LANCZOS_GRAPH)
    bool rsvd          = false;  // randomized SVD for truncation (DMRG_GPU_OPT_RSVD)
    bool sparse_mpo    = false;  // block-sparse apply_heff (DMRG_GPU_OPT_SPARSE_MPO)
    bool fuse_lanczos  = false;  // fused axpy+reorth kernel (DMRG_GPU_OPT_FUSE_LANCZOS)
    bool d_pad         = false;  // pad D_mpo to MFMA-friendly size (DMRG_GPU_OPT_D_PAD)

    bool profile       = false;  // DMRG_GPU_PROFILE=1 enables per-phase hipEvent timing

    static bool env_flag(const char* name) {
        const char* v = std::getenv(name);
        if (!v) return false;
        return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
    }
    void load_from_env() {
        device_k      = env_flag("DMRG_GPU_OPT_DEVICE_K");
        lanczos_fixed = env_flag("DMRG_GPU_OPT_LANCZOS_FIXED");
        lanczos_graph = env_flag("DMRG_GPU_OPT_LANCZOS_GRAPH");
        rsvd          = env_flag("DMRG_GPU_OPT_RSVD");
        sparse_mpo    = env_flag("DMRG_GPU_OPT_SPARSE_MPO");
        fuse_lanczos  = env_flag("DMRG_GPU_OPT_FUSE_LANCZOS");
        d_pad         = env_flag("DMRG_GPU_OPT_D_PAD");
        profile       = env_flag("DMRG_GPU_PROFILE");
    }
    void print(FILE* out) const {
        std::fprintf(out, "== GPU options ==\n");
        std::fprintf(out, "  device_k       : %s\n", device_k      ? "on" : "off");
        std::fprintf(out, "  lanczos_fixed  : %s\n", lanczos_fixed ? "on" : "off");
        std::fprintf(out, "  lanczos_graph  : %s\n", lanczos_graph ? "on" : "off");
        std::fprintf(out, "  rsvd           : %s\n", rsvd          ? "on" : "off");
        std::fprintf(out, "  sparse_mpo     : %s\n", sparse_mpo    ? "on" : "off");
        std::fprintf(out, "  fuse_lanczos   : %s\n", fuse_lanczos  ? "on" : "off");
        std::fprintf(out, "  d_pad          : %s\n", d_pad         ? "on" : "off");
        std::fprintf(out, "  profile        : %s\n", profile       ? "on" : "off");
    }
};

// ============================================================================
// PhaseTimer — lightweight hipEvent accumulator. Records (start, stop) event
// pairs during the run; defers elapsed-time computation to report() so the
// hot path never synchronizes. Disabled unless GpuOpts::profile is on.
// ============================================================================
struct PhaseTimer {
    const char* name = "";
    std::vector<hipEvent_t> starts;
    std::vector<hipEvent_t> stops;
    bool enabled = false;

    void init(const char* n, bool en) { name = n; enabled = en; }
    void begin(hipStream_t s) {
        if (!enabled) return;
        hipEvent_t e;
        hipEventCreate(&e);
        hipEventRecord(e, s);
        starts.push_back(e);
    }
    void end(hipStream_t s) {
        if (!enabled) return;
        hipEvent_t e;
        hipEventCreate(&e);
        hipEventRecord(e, s);
        stops.push_back(e);
    }
    double total_ms() {
        double sum = 0.0;
        for (size_t i = 0; i < starts.size() && i < stops.size(); i++) {
            hipEventSynchronize(stops[i]);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, starts[i], stops[i]);
            sum += (double)ms;
        }
        return sum;
    }
    int calls() const { return (int)stops.size(); }
    ~PhaseTimer() {
        for (auto e : starts) hipEventDestroy(e);
        for (auto e : stops)  hipEventDestroy(e);
    }
};

#endif // GPU_OPTS_H
