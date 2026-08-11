// conn_fast.cpp -- compiled lookahead for the AutoConnect bot (issue: deadlocks).
//
// Three pybind11 exports (drop-in accelerators for solver/conn.py + bot.py):
//   connectable_pairs(present)        -> (N,4) int32  every <=2-turn pair
//   rollout_clear_rate(present,sim,thr,K,seed) -> double  Monte-Carlo clear prob
//   solvable(present,sim,thr,max_depth,topk)   -> bool    exact (endgame) search
//
// Connectivity is a faithful, count-validated port of solver/conn.py (matches
// Python's all_connectable_pairs_anylabel exactly: 324/484/676 on full 8x12 /
// 10x14 / 12x16 boards). Rollouts make deep deadlock risk measurable without
// the exponential cost of exact search.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// connectivity primitives (padded passable map; 1 = empty/passable, border = 1)
// ---------------------------------------------------------------------------
static inline bool h_clear(const uint8_t *p, int pc, int r, int c1, int c2) {
    if (c1 == c2) return true;
    int a = c1 < c2 ? c1 : c2, b = c1 < c2 ? c2 : c1;
    for (int c = a + 1; c < b; c++)
        if (!p[r * pc + c]) return false;
    return true;
}
static inline bool v_clear(const uint8_t *p, int pc, int c, int r1, int r2) {
    if (r1 == r2) return true;
    int a = r1 < r2 ? r1 : r2, b = r1 < r2 ? r2 : r1;
    for (int r = a + 1; r < b; r++)
        if (!p[r * pc + c]) return false;
    return true;
}
static inline bool connectable(const uint8_t *p, int PR, int PC,
                               int r1, int c1, int r2, int c2) {
    if (r1 == r2 && h_clear(p, PC, r1, c1, c2)) return true;
    if (c1 == c2 && v_clear(p, PC, c1, r1, r2)) return true;
    if (p[r1 * PC + c2] && h_clear(p, PC, r1, c1, c2) && v_clear(p, PC, c2, r1, r2)) return true;
    if (p[r2 * PC + c1] && v_clear(p, PC, c1, r1, r2) && h_clear(p, PC, r2, c1, c2)) return true;
    for (int R_ = 0; R_ < PR; R_++)
        if (p[R_ * PC + c1] && p[R_ * PC + c2] &&
            v_clear(p, PC, c1, r1, R_) && h_clear(p, PC, R_, c1, c2) && v_clear(p, PC, c2, R_, r2))
            return true;
    for (int C_ = 0; C_ < PC; C_++)
        if (p[r1 * PC + C_] && p[r2 * PC + C_] &&
            h_clear(p, PC, r1, c1, C_) && v_clear(p, PC, C_, r1, r2) && h_clear(p, PC, r2, c2, C_))
            return true;
    return false;
}

// Build the padded passable map (PR*PC) from a present mask (R*C, 1=present).
static inline void build_passable(const uint8_t *present, int R, int C, std::vector<uint8_t> &p) {
    int PR = R + 2, PC = C + 2;
    p.assign((size_t)PR * PC, 1);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (present[r * C + c]) p[((size_t)r + 1) * PC + (c + 1)] = 0;
}

// collect present cells as padded coords
static inline void present_cells(const uint8_t *present, int R, int C,
                                 std::vector<int> &rc, std::vector<int> &cc) {
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (present[r * C + c]) { rc.push_back(r + 1); cc.push_back(c + 1); }
}

// ---------------------------------------------------------------------------
// sim indexing: sim is (R,C,R,C) float, row-major -> flat index
// ---------------------------------------------------------------------------
static inline size_t sim_idx(int R, int C, int r1, int c1, int r2, int c2) {
    return ((size_t)(r1 * C + c1) * (R * C) + (r2 * C + c2));
}

// ===========================================================================
// 1) connectable_pairs
// ===========================================================================
static py::array_t<int32_t> py_connectable_pairs(py::array_t<bool> present_arr) {
    auto bp = present_arr.request();
    if (bp.ndim != 2) throw std::runtime_error("present must be 2D");
    int R = (int)bp.shape[0], C = (int)bp.shape[1];
    const uint8_t *present = (const uint8_t *)bp.ptr;

    std::vector<uint8_t> p;
    build_passable(present, R, C, p);
    int PR = R + 2, PC = C + 2;
    std::vector<int> rc, cc;
    present_cells(present, R, C, rc, cc);

    std::vector<int32_t> out;
    out.reserve(rc.size() * rc.size() / 2);
    for (size_t i = 0; i < rc.size(); i++)
        for (size_t j = i + 1; j < rc.size(); j++)
            if (connectable(p.data(), PR, PC, rc[i], cc[i], rc[j], cc[j])) {
                out.push_back(rc[i] - 1); out.push_back(cc[i] - 1);
                out.push_back(rc[j] - 1); out.push_back(cc[j] - 1);
            }
    size_t N = out.size() / 4;
    auto result = py::array_t<int32_t>({(py::ssize_t)N, (py::ssize_t)4});
    std::copy(out.begin(), out.end(), (int32_t *)result.request().ptr);
    return result;
}

// ===========================================================================
// rollout helpers
// ===========================================================================
// Pick a uniformly random same-type connectable pair (reservoir sampled).
// Returns false if none exists. cur is the present mask (1=present), mutated by caller.
static inline bool random_same_pair(const uint8_t *cur, int R, int C,
                                    const float *sim, float thr, std::mt19937 &rng,
                                    int &r1o, int &c1o, int &r2o, int &c2o) {
    std::vector<uint8_t> p;
    build_passable(cur, R, C, p);
    int PR = R + 2, PC = C + 2;
    std::vector<int> rc, cc;
    present_cells(cur, R, C, rc, cc);
    int count = 0;
    bool have = false;
    for (size_t i = 0; i < rc.size(); i++) {
        for (size_t j = i + 1; j < rc.size(); j++) {
            if (sim[sim_idx(R, C, rc[i] - 1, cc[i] - 1, rc[j] - 1, cc[j] - 1)] < thr) continue;
            if (!connectable(p.data(), PR, PC, rc[i], cc[i], rc[j], cc[j])) continue;
            ++count;
            if (rng() % (unsigned)count == 0) {
                r1o = rc[i] - 1; c1o = cc[i] - 1; r2o = rc[j] - 1; c2o = cc[j] - 1;
                have = true;
            }
        }
    }
    return have;
}

// ===========================================================================
// 2) rollout_clear_rate
// ===========================================================================
static double py_rollout_clear_rate(py::array_t<bool> present_arr,
                                    py::array_t<float> sim_arr,
                                    double thr, int K, int seed) {
    auto bp = present_arr.request();
    auto bs = sim_arr.request();
    if (bp.ndim != 2) throw std::runtime_error("present must be 2D");
    int R = (int)bp.shape[0], C = (int)bp.shape[1];
    const uint8_t *present0 = (const uint8_t *)bp.ptr;
    const float *sim = (const float *)bs.ptr;
    if (K <= 0) K = 1;
    std::mt19937 rng((unsigned)seed);

    std::vector<uint8_t> cur((size_t)R * C);
    int init_left = 0;
    for (int i = 0; i < R * C; i++) init_left += present0[i] ? 1 : 0;
    float fthr = (float)thr;
    int success = 0;

    for (int k = 0; k < K; k++) {
        std::copy(present0, present0 + (size_t)R * C, cur.begin());
        int left = init_left;
        bool cleared = (left == 0);
        while (!cleared) {
            int r1, c1, r2, c2;
            if (!random_same_pair(cur.data(), R, C, sim, fthr, rng, r1, c1, r2, c2)) break;
            cur[(size_t)r1 * C + c1] = 0;
            cur[(size_t)r2 * C + c2] = 0;
            left -= 2;
            cleared = (left <= 0);
        }
        if (cleared) ++success;
    }
    return (double)success / (double)K;
}

// ===========================================================================
// 2b) rollout_mean_steps -- mean removal-steps per rollout (whether it clears
//     or deadlocks). Predicts rollout TIME (per-step cost is constant), i.e.
//     the rollout-length inflation from a soft/permissive backbone.
// ===========================================================================
static double py_rollout_mean_steps(py::array_t<bool> present_arr,
                                    py::array_t<float> sim_arr,
                                    double thr, int K, int seed) {
    auto bp = present_arr.request();
    auto bs = sim_arr.request();
    if (bp.ndim != 2) throw std::runtime_error("present must be 2D");
    int R = (int)bp.shape[0], C = (int)bp.shape[1];
    const uint8_t *present0 = (const uint8_t *)bp.ptr;
    const float *sim = (const float *)bs.ptr;
    if (K <= 0) K = 1;
    std::mt19937 rng((unsigned)seed);
    std::vector<uint8_t> cur((size_t)R * C);
    int init_left = 0;
    for (int i = 0; i < R * C; i++) init_left += present0[i] ? 1 : 0;
    float fthr = (float)thr;
    long total_steps = 0;
    for (int k = 0; k < K; k++) {
        std::copy(present0, present0 + (size_t)R * C, cur.begin());
        int left = init_left, steps = 0;
        while (left > 0) {
            int r1, c1, r2, c2;
            if (!random_same_pair(cur.data(), R, C, sim, fthr, rng, r1, c1, r2, c2)) break;
            cur[(size_t)r1 * C + c1] = 0;
            cur[(size_t)r2 * C + c2] = 0;
            left -= 2;
            ++steps;
        }
        total_steps += steps;
    }
    return (double)total_steps / (double)K;
}

// ===========================================================================
// 3) solvable (exact, endgame) -- port of bot.py _solvable
// ===========================================================================
static bool solvable_rec(uint8_t *cur, int R, int C, const float *sim, float thr,
                         int max_depth, int topk, int depth,
                         std::unordered_map<std::string, bool> &memo) {
    int left = 0;
    for (int i = 0; i < R * C; i++) left += cur[i];
    if (left == 0) return true;
    if (left & 1) return false;            // odd count can't clear
    if (depth > max_depth) return true;    // bail: assume solvable (avoid blowup)
    std::string key((const char *)cur, (size_t)R * C);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    std::vector<uint8_t> p;
    build_passable(cur, R, C, p);
    int PR = R + 2, PC = C + 2;
    std::vector<int> rc, cc;
    present_cells(cur, R, C, rc, cc);

    std::vector<std::pair<float, uint32_t>> cands;  // (sim score, packed pair)
    for (size_t i = 0; i < rc.size(); i++)
        for (size_t j = i + 1; j < rc.size(); j++) {
            float v = sim[sim_idx(R, C, rc[i] - 1, cc[i] - 1, rc[j] - 1, cc[j] - 1)];
            if (v < thr) continue;
            if (!connectable(p.data(), PR, PC, rc[i], cc[i], rc[j], cc[j])) continue;
            uint32_t pack = ((uint32_t)(rc[i] - 1) << 24) | ((uint32_t)(cc[i] - 1) << 16) |
                            ((uint32_t)(rc[j] - 1) << 8) | (uint32_t)(cc[j] - 1);
            cands.push_back({v, pack});
        }
    std::sort(cands.begin(), cands.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    int take = std::min((int)cands.size(), topk);
    for (int t = 0; t < take; t++) {
        uint32_t pack = cands[t].second;
        int r1 = (pack >> 24) & 0xFF, c1 = (pack >> 16) & 0xFF;
        int r2 = (pack >> 8) & 0xFF, c2 = pack & 0xFF;
        cur[(size_t)r1 * C + c1] = 0;
        cur[(size_t)r2 * C + c2] = 0;
        bool ok = solvable_rec(cur, R, C, sim, thr, max_depth, topk, depth + 1, memo);
        cur[(size_t)r1 * C + c1] = 1;  // restore (the cell was present)
        cur[(size_t)r2 * C + c2] = 1;
        if (ok) { memo[key] = true; return true; }
    }
    memo[key] = false;
    return false;
}

static bool py_solvable(py::array_t<bool> present_arr, py::array_t<float> sim_arr,
                        double thr, int max_depth, int topk) {
    auto bp = present_arr.request();
    auto bs = sim_arr.request();
    if (bp.ndim != 2) throw std::runtime_error("present must be 2D");
    int R = (int)bp.shape[0], C = (int)bp.shape[1];
    std::vector<uint8_t> cur((const uint8_t *)bp.ptr, (const uint8_t *)bp.ptr + (size_t)R * C);
    std::unordered_map<std::string, bool> memo;
    return solvable_rec(cur.data(), R, C, (const float *)bs.ptr, (float)thr,
                        max_depth <= 0 ? 1000000 : max_depth, topk <= 0 ? 12 : topk, 0, memo);
}

// ===========================================================================
PYBIND11_MODULE(conn_fast, m) {
    m.doc() = "Compiled lookahead: connectivity, Monte-Carlo rollouts, exact endgame.";
    m.def("connectable_pairs", &py_connectable_pairs,
          py::arg("present"),
          "Every <=2-turn connectable pair as (N,4) int32 [r1,c1,r2,c2].");
    m.def("rollout_clear_rate", &py_rollout_clear_rate,
          py::arg("present"), py::arg("sim"), py::arg("thr"),
          py::arg("K"), py::arg("seed"),
          "Monte-Carlo clear probability: K random same-type completions.");
    m.def("rollout_mean_steps", &py_rollout_mean_steps,
          py::arg("present"), py::arg("sim"), py::arg("thr"),
          py::arg("K"), py::arg("seed"),
          "Mean removal-steps per rollout (predicts rollout time).");
    m.def("solvable", &py_solvable,
          py::arg("present"), py::arg("sim"), py::arg("thr"),
          py::arg("max_depth") = 0, py::arg("topk") = 0,
          "Exact solvability (endgame): can present be fully cleared?");
}
