#include "keel/interp.h"

#include "keel/arena.h"
#include "keel/math.h"

namespace keel::interp {

bool aveknt(ArrayRef<f64> t, i32 k, MutArrayRef<f64> out) noexcept {
    const usize n = t.size();
    if (k < 2 || as<usize>(k) > n) return false;
    const usize m = n - as<usize>(k);
    if (out.size() < m) return false;
    if (m == 0) return true;
    const f64 inv = 1.0 / as<f64>(k - 1);
    const usize ku = as<usize>(k);
    for (usize i = 0; i < m; ++i) {
        f64 sum = 0.0;
        for (usize j = 1; j < ku; ++j) sum += t[i + j];
        out[i] = sum * inv;
    }
    return true;
}

bool augknt(ArrayRef<f64> knots, i32 k, MutArrayRef<f64> out) noexcept {
    const usize n = knots.size();
    if (k < 1 || n == 0) return false;
    const usize ku = as<usize>(k);
    const usize out_n = n + 2 * (ku - 1);
    if (out.size() < out_n) return false;
    const f64 first = knots[0];
    const f64 last = knots[n - 1];
    usize idx = 0;
    for (usize i = 0; i < ku; ++i) out[idx++] = first;
    for (usize i = 1; i + 1 < n; ++i) out[idx++] = knots[i];
    for (usize i = 0; i < ku; ++i) out[idx++] = last;
    return true;
}

bool brk2knt(ArrayRef<f64> breaks, ArrayRef<usize> mults, Array<f64>& out_knots,
             Allocator* alloc) {
    const usize n = breaks.size();
    if (mults.size() != n) return false;
    usize total = 0;
    for (usize i = 0; i < n; ++i) total += mults[i];
    out_knots = Array<f64>(alloc);
    out_knots.resize(total);
    usize idx = 0;
    for (usize i = 0; i < n; ++i) {
        const usize mi = mults[i];
        const f64 bi = breaks[i];
        for (usize j = 0; j < mi; ++j) out_knots[idx++] = bi;
    }
    return true;
}

void knt2brk(ArrayRef<f64> knots, Array<f64>& out_breaks, Array<usize>& out_mults,
             Allocator* alloc) {
    out_breaks = Array<f64>(alloc);
    out_mults = Array<usize>(alloc);
    const usize n = knots.size();
    if (n == 0) return;
    f64 cur = knots[0];
    usize cnt = 1;
    for (usize i = 1; i < n; ++i) {
        const f64 ki = knots[i];
        if (ki == cur) {
            ++cnt;
        } else {
            out_breaks.push_back(cur);
            out_mults.push_back(cnt);
            cur = ki;
            cnt = 1;
        }
    }
    out_breaks.push_back(cur);
    out_mults.push_back(cnt);
}

bool ppval_nd(ArrayRef<f64> breaks, ArrayRef<f64> coefs_colmajor, usize pieces, usize order,
              usize dim, ArrayRef<f64> xq, MutArrayRef<f64> out_colmajor) noexcept {
    if (pieces == 0 || order == 0 || dim == 0) return false;
    const usize rows = dim * pieces;
    if (breaks.size() < pieces + 1 || coefs_colmajor.size() < rows * order ||
        out_colmajor.size() < dim * xq.size()) {
        return false;
    }
    const usize nq = xq.size();
    for (usize i = 0; i < nq; ++i) {
        const f64 xi = xq[i];
        if (is_nan(xi)) {
            for (usize r = 0; r < dim; ++r) {
                out_colmajor[r + i * dim] = Limits<f64>::quiet_nan();
            }
            continue;
        }
        usize j = 0;
        if (xi <= breaks[0]) {
            j = 0;
        } else if (xi >= breaks[pieces]) {
            j = pieces - 1;
        } else {
            usize lo = 0;
            usize hi = pieces - 1;
            while (lo < hi) {
                const usize mid = lo + (hi - lo + 1) / 2;
                if (breaks[mid] <= xi) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            j = lo;
        }
        const f64 u = xi - breaks[j];
        for (usize r = 0; r < dim; ++r) {
            const usize row = r + j * dim;
            f64 y = coefs_colmajor[row];
            for (usize m = 1; m < order; ++m) {
                y = y * u + coefs_colmajor[row + m * rows];
            }
            out_colmajor[r + i * dim] = y;
        }
    }
    return true;
}

bool pp_deriv(ArrayRef<f64> coefs_colmajor, usize pieces, usize order, usize dim, i32 deriv_order,
              Array<f64>& out_coefs_colmajor, usize& out_order, Allocator* alloc) {
    if (pieces == 0 || order == 0 || dim == 0 || deriv_order < 0) return false;
    const usize rows = dim * pieces;
    if (coefs_colmajor.size() < rows * order) return false;

    if (deriv_order == 0) {
        out_order = order;
        out_coefs_colmajor = Array<f64>(rows * order, alloc);
        for (usize i = 0; i < rows * order; ++i) out_coefs_colmajor[i] = coefs_colmajor[i];
        return true;
    }
    if (as<usize>(deriv_order) >= order) {
        out_order = 1;
        out_coefs_colmajor = Array<f64>(rows, 0.0, alloc);
        return true;
    }

    const usize d_ord = as<usize>(deriv_order);
    out_order = order - d_ord;
    out_coefs_colmajor = Array<f64>(rows * out_order, 0.0, alloc);
    for (usize m = 0; m < out_order; ++m) {
        // Falling factorial: (order - 1 - m) * ... * (order - m - d_ord)
        f64 factor = 1.0;
        const usize deg = order - 1 - m;
        for (usize s = 0; s < d_ord; ++s) {
            factor *= as<f64>(deg - s);
        }
        for (usize r = 0; r < rows; ++r) {
            out_coefs_colmajor[r + m * rows] = coefs_colmajor[r + m * rows] * factor;
        }
    }
    return true;
}

bool pp_integ(ArrayRef<f64> breaks, ArrayRef<f64> coefs_colmajor, usize pieces, usize order,
              usize dim, MutArrayRef<f64> out_coefs_colmajor) noexcept {
    if (pieces == 0 || order == 0 || dim == 0) return false;
    const usize rows = dim * pieces;
    const usize new_order = order + 1;
    if (breaks.size() < pieces + 1 || coefs_colmajor.size() < rows * order ||
        out_coefs_colmajor.size() < rows * new_order) {
        return false;
    }
    for (usize r = 0; r < dim; ++r) {
        f64 prev_const = 0.0;
        for (usize j = 0; j < pieces; ++j) {
            const usize row = r + j * dim;
            for (usize m = 0; m < order; ++m) {
                out_coefs_colmajor[row + m * rows] =
                    coefs_colmajor[row + m * rows] / as<f64>(order - m);
            }
            out_coefs_colmajor[row + order * rows] = prev_const;
            const f64 h = breaks[j + 1] - breaks[j];
            f64 y = out_coefs_colmajor[row];
            for (usize m = 1; m < order; ++m) {
                y = y * h + out_coefs_colmajor[row + m * rows];
            }
            prev_const += y * h;
        }
    }
    return true;
}

} // namespace keel::interp
