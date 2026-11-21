#include "keel/interp.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::interp {

namespace {

usize find_interval(const f64* x, usize n, f64 xq) noexcept {
    if (xq <= x[0]) return 0;
    if (xq >= x[n - 1]) return n - 2;
    // Upper bound binary search: first index with x[idx] > xq.
    usize lo = 0;
    usize hi = n;
    while (lo < hi) {
        const usize mid = lo + (hi - lo) / 2;
        if (x[mid] <= xq) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo == 0) ? 0 : (lo - 1);
}

usize find_cell(const f64* grid, usize n, f64 q) noexcept {
    if (n < 2 || is_nan(q) || q < grid[0] || q > grid[n - 1]) return kInvalidIndex;
    usize lo = 0;
    usize hi = n - 1;
    while (hi - lo > 1) {
        const usize mid = lo + (hi - lo) / 2;
        if (grid[mid] <= q) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

f64 keys_cubic(f64 s) noexcept {
    s = keel::abs(s);
    if (s <= 1.0) return ((1.5 * s - 2.5) * s) * s + 1.0;
    if (s < 2.0) return (((-0.5 * s + 2.5) * s) - 4.0) * s + 2.0;
    return 0.0;
}

// Not-a-knot cubic spline second derivatives sigma[0..n-1].
void compute_spline_sigma(const f64* y, usize n, const f64* h, f64* sigma, Allocator* salloc) {
    for (usize i = 0; i < n; ++i) sigma[i] = 0.0;
    if (n < 3) return;

    if (n == 3) {
        const f64 c = 2.0 * ((y[2] - y[1]) / h[1] - (y[1] - y[0]) / h[0]) / (h[0] + h[1]);
        sigma[0] = c;
        sigma[1] = c;
        sigma[2] = c;
        return;
    }

    const usize m = n - 2;
    Array<f64> diag(m, salloc);
    Array<f64> upper(m, 0.0, salloc);
    Array<f64> lower(m, 0.0, salloc);
    Array<f64> rhs(m, salloc);

    for (usize i = 0; i < m; ++i) {
        const usize j = i + 1;
        diag[i] = 2.0 * (h[j - 1] + h[j]);
        rhs[i] = 6.0 * ((y[j + 1] - y[j]) / h[j] - (y[j] - y[j - 1]) / h[j - 1]);
        if (i > 0) lower[i] = h[j - 1];
        if (i < m - 1) upper[i] = h[j];
    }

    if (m >= 2) {
        const f64 h0 = h[0];
        const f64 h1 = h[1];
        diag[0] = (h0 + h1) * (h0 + 2.0 * h1) / h1;
        upper[0] = (h1 * h1 - h0 * h0) / h1;

        const f64 hl = h[n - 3];
        const f64 hl1 = h[n - 2];
        diag[m - 1] = (hl + hl1) * (2.0 * hl + hl1) / hl;
        lower[m - 1] = (hl * hl - hl1 * hl1) / hl;
    }

    for (usize i = 1; i < m; ++i) {
        const f64 w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    sigma[m] = rhs[m - 1] / diag[m - 1];
    for (usize i = m - 1; i-- > 0;) {
        sigma[i + 1] = (rhs[i] - upper[i] * sigma[i + 2]) / diag[i];
    }

    if (m >= 2) {
        const f64 h0 = h[0];
        const f64 h1 = h[1];
        sigma[0] = ((h0 + h1) * sigma[1] - h0 * sigma[2]) / h1;
        const f64 hl = h[n - 3];
        const f64 hl1 = h[n - 2];
        sigma[n - 1] = ((hl + hl1) * sigma[n - 2] - hl1 * sigma[n - 3]) / hl;
    }
}

void compute_pchip_slopes(const f64* h, const f64* delta, usize n, f64* d) noexcept {
    const usize nm1 = n - 1;
    if (n == 2) {
        d[0] = delta[0];
        d[1] = delta[0];
        return;
    }
    for (usize i = 1; i < nm1; ++i) {
        if (delta[i - 1] * delta[i] <= 0.0) {
            d[i] = 0.0;
        } else {
            const f64 w1 = 2.0 * h[i] + h[i - 1];
            const f64 w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
    if (d[0] * delta[0] < 0.0) {
        d[0] = 0.0;
    } else if (delta[0] * delta[1] < 0.0 && keel::abs(d[0]) > keel::abs(3.0 * delta[0])) {
        d[0] = 3.0 * delta[0];
    }

    d[nm1] = ((2.0 * h[nm1 - 1] + h[nm1 - 2]) * delta[nm1 - 1] - h[nm1 - 1] * delta[nm1 - 2]) /
             (h[nm1 - 1] + h[nm1 - 2]);
    if (d[nm1] * delta[nm1 - 1] < 0.0) {
        d[nm1] = 0.0;
    } else if (delta[nm1 - 2] * delta[nm1 - 1] < 0.0 &&
               keel::abs(d[nm1]) > keel::abs(3.0 * delta[nm1 - 1])) {
        d[nm1] = 3.0 * delta[nm1 - 1];
    }
}

void compute_makima_slopes(const f64* delta, usize n, f64* d, Allocator* salloc) {
    const usize nm1 = n - 1;
    if (n == 2) {
        d[0] = delta[0];
        d[1] = delta[0];
        return;
    }
    Array<f64> m_ext(n + 3, salloc);
    for (usize i = 0; i < nm1; ++i) m_ext[2 + i] = delta[i];
    m_ext[1] = 2.0 * m_ext[2] - m_ext[3];
    m_ext[0] = 2.0 * m_ext[1] - m_ext[2];
    m_ext[2 + nm1] = 2.0 * m_ext[2 + nm1 - 1] - m_ext[2 + nm1 - 2];
    m_ext[2 + nm1 + 1] = 2.0 * m_ext[2 + nm1] - m_ext[2 + nm1 - 1];

    for (usize i = 0; i < n; ++i) {
        const f64 ml2 = m_ext[i];
        const f64 ml1 = m_ext[i + 1];
        const f64 mr1 = m_ext[i + 2];
        const f64 mr2 = m_ext[i + 3];
        const f64 w1 = keel::abs(mr2 - mr1) + 0.5 * keel::abs(mr2 + mr1);
        const f64 w2 = keel::abs(ml1 - ml2) + 0.5 * keel::abs(ml1 + ml2);
        const f64 wsum = w1 + w2;
        d[i] = (wsum == 0.0) ? 0.0 : (w1 * ml1 + w2 * mr1) / wsum;
    }
}

void eval_linear(const f64* x, const f64* y, usize n, const f64* xq, usize nq, f64* out) noexcept {
    for (usize k = 0; k < nq; ++k) {
        if (is_nan(xq[k])) {
            out[k] = Limits<f64>::quiet_nan();
            continue;
        }
        const usize i = find_interval(x, n, xq[k]);
        const f64 dx = x[i + 1] - x[i];
        if (dx == 0.0) {
            out[k] = y[i];
        } else {
            const f64 t = (xq[k] - x[i]) / dx;
            out[k] = y[i] + t * (y[i + 1] - y[i]);
        }
    }
}

void eval_spline(const f64* x, const f64* y, usize n, const f64* xq, usize nq, f64* out,
                 Allocator* salloc) {
    if (n < 3) {
        eval_linear(x, y, n, xq, nq, out);
        return;
    }
    const usize nm1 = n - 1;
    Array<f64> h(nm1, salloc);
    for (usize i = 0; i < nm1; ++i) h[i] = x[i + 1] - x[i];
    Array<f64> sigma(n, salloc);
    compute_spline_sigma(y, n, h.data(), sigma.data(), salloc);

    for (usize k = 0; k < nq; ++k) {
        if (is_nan(xq[k])) {
            out[k] = Limits<f64>::quiet_nan();
            continue;
        }
        const usize i = find_interval(x, n, xq[k]);
        const f64 dx = xq[k] - x[i];
        const f64 dx1 = x[i + 1] - xq[k];
        const f64 hi = h[i];
        out[k] = sigma[i] * dx1 * dx1 * dx1 / (6.0 * hi) +
                 sigma[i + 1] * dx * dx * dx / (6.0 * hi) +
                 (y[i] / hi - sigma[i] * hi / 6.0) * dx1 +
                 (y[i + 1] / hi - sigma[i + 1] * hi / 6.0) * dx;
    }
}

void eval_hermite(const f64* x, const f64* y, const f64* h, const f64* d, usize n, const f64* xq,
                  usize nq, f64* out) noexcept {
    for (usize k = 0; k < nq; ++k) {
        if (is_nan(xq[k])) {
            out[k] = Limits<f64>::quiet_nan();
            continue;
        }
        const usize i = find_interval(x, n, xq[k]);
        const f64 t = (xq[k] - x[i]) / h[i];
        const f64 t2 = t * t;
        const f64 t3 = t2 * t;
        const f64 h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const f64 h10 = t3 - 2.0 * t2 + t;
        const f64 h01 = -2.0 * t3 + 3.0 * t2;
        const f64 h11 = t3 - t2;
        out[k] = h00 * y[i] + h10 * h[i] * d[i] + h01 * y[i + 1] + h11 * h[i] * d[i + 1];
    }
}

void build_vpad_2d(const f64* v, usize rows, usize cols, Array<f64>& vpad) {
    const usize pr = rows + 2;
    const usize pc = cols + 2;
    vpad.assign(pr * pc, 0.0);
    auto at = [&](usize i, usize j) -> f64& { return vpad[j * pr + i]; };
    for (usize j = 0; j < cols; ++j) {
        for (usize i = 0; i < rows; ++i) {
            at(i + 1, j + 1) = v[j * rows + i];
        }
    }
    for (usize j = 0; j < cols; ++j) {
        const f64* col = v + j * rows;
        at(0, j + 1) = 3.0 * col[0] - 3.0 * col[1] + col[2];
        at(rows + 1, j + 1) = 3.0 * col[rows - 1] - 3.0 * col[rows - 2] + col[rows - 3];
    }
    for (usize i = 0; i < pr; ++i) {
        at(i, 0) = 3.0 * at(i, 1) - 3.0 * at(i, 2) + at(i, 3);
        at(i, cols + 1) = 3.0 * at(i, cols) - 3.0 * at(i, cols - 1) + at(i, cols - 2);
    }
}

f64 sample_2d_basic(const f64* v, const f64* vpad, usize rows, usize cols, const f64* x_grid,
                    const f64* y_grid, f64 xq, f64 yq, Interp2Method method) noexcept {
    const usize ix = find_cell(x_grid, cols, xq);
    const usize iy = find_cell(y_grid, rows, yq);
    if (ix == kInvalidIndex || iy == kInvalidIndex) return Limits<f64>::quiet_nan();

    if (method == Interp2Method::Cubic) {
        const f64 tx = (xq - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix]);
        const f64 ty = (yq - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy]);
        const f64 wx[4] = {keys_cubic(1.0 + tx), keys_cubic(tx), keys_cubic(1.0 - tx),
                           keys_cubic(2.0 - tx)};
        const f64 wy[4] = {keys_cubic(1.0 + ty), keys_cubic(ty), keys_cubic(1.0 - ty),
                           keys_cubic(2.0 - ty)};
        const usize pr = rows + 2;
        f64 acc = 0.0;
        for (usize a = 0; a < 4; ++a) {
            f64 row_acc = 0.0;
            for (usize b = 0; b < 4; ++b) {
                row_acc += wx[b] * vpad[(ix + b) * pr + (iy + a)];
            }
            acc += wy[a] * row_acc;
        }
        return acc;
    }

    if (method == Interp2Method::Nearest) {
        const usize cx = (xq - x_grid[ix] <= x_grid[ix + 1] - xq) ? ix : (ix + 1);
        const usize cy = (yq - y_grid[iy] <= y_grid[iy + 1] - yq) ? iy : (iy + 1);
        return v[cx * rows + cy];
    }

    const f64 tx = (xq - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix]);
    const f64 ty = (yq - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy]);
    const f64 v00 = v[ix * rows + iy];
    const f64 v10 = v[(ix + 1) * rows + iy];
    const f64 v01 = v[ix * rows + (iy + 1)];
    const f64 v11 = v[(ix + 1) * rows + (iy + 1)];
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 +
           tx * ty * v11;
}

} // namespace

// ============================================================================
// 1. Grid Validation Helpers
// ============================================================================

bool is_strictly_increasing(ArrayRef<f64> grid) noexcept {
    for (usize i = 1; i < grid.size(); ++i) {
        if (!(grid[i] > grid[i - 1])) return false;
    }
    return true;
}

bool is_uniform_grid(ArrayRef<f64> grid) noexcept {
    if (grid.size() < 2) return true;
    const f64 step = grid[1] - grid[0];
    const f64 tol = 1e-10 * keel::max(1.0, keel::abs(step));
    for (usize i = 2; i < grid.size(); ++i) {
        if (keel::abs((grid[i] - grid[i - 1]) - step) > tol) return false;
    }
    return true;
}

// ============================================================================
// 2. 1D Interpolation
// ============================================================================

bool interp1(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> xq, MutArrayRef<f64> out,
             Interp1Method method, ExtrapMode extrap, f64 extrap_val, Allocator* scratch_alloc) {
    const usize n = x.size();
    const usize nq = xq.size();
    if (n < 2 || y.size() != n || out.size() < nq) return false;
    if (nq == 0) return true;

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();
    const f64 nan = Limits<f64>::quiet_nan();

    switch (method) {
        case Interp1Method::Linear:
            eval_linear(x.data(), y.data(), n, xq.data(), nq, out.data());
            break;
        case Interp1Method::Nearest:
            for (usize k = 0; k < nq; ++k) {
                if (is_nan(xq[k])) {
                    out[k] = nan;
                    continue;
                }
                const usize i = find_interval(x.data(), n, xq[k]);
                out[k] = (keel::abs(xq[k] - x[i]) < keel::abs(xq[k] - x[i + 1])) ? y[i] : y[i + 1];
            }
            break;
        case Interp1Method::Previous:
            for (usize k = 0; k < nq; ++k) {
                const f64 q = xq[k];
                if (is_nan(q) || q < x[0] || q > x[n - 1]) {
                    out[k] = nan;
                    continue;
                }
                if (q == x[n - 1]) {
                    out[k] = y[n - 1];
                } else {
                    out[k] = y[find_interval(x.data(), n, q)];
                }
            }
            break;
        case Interp1Method::Next:
            for (usize k = 0; k < nq; ++k) {
                const f64 q = xq[k];
                if (is_nan(q) || q < x[0] || q > x[n - 1]) {
                    out[k] = nan;
                    continue;
                }
                // Lower bound binary search: first index with x[idx] >= q.
                usize lo = 0;
                usize hi = n - 1;
                while (lo < hi) {
                    const usize mid = lo + (hi - lo) / 2;
                    if (x[mid] < q) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                out[k] = y[lo];
            }
            break;
        case Interp1Method::Spline:
            eval_spline(x.data(), y.data(), n, xq.data(), nq, out.data(), salloc);
            break;
        case Interp1Method::Pchip:
            if (n < 3) {
                eval_linear(x.data(), y.data(), n, xq.data(), nq, out.data());
            } else {
                const usize nm1 = n - 1;
                Array<f64> h(nm1, salloc);
                Array<f64> delta(nm1, salloc);
                for (usize i = 0; i < nm1; ++i) {
                    h[i] = x[i + 1] - x[i];
                    delta[i] = (y[i + 1] - y[i]) / h[i];
                }
                Array<f64> d(n, 0.0, salloc);
                compute_pchip_slopes(h.data(), delta.data(), n, d.data());
                eval_hermite(x.data(), y.data(), h.data(), d.data(), n, xq.data(), nq, out.data());
            }
            break;
        case Interp1Method::Makima:
            if (n < 3) {
                eval_linear(x.data(), y.data(), n, xq.data(), nq, out.data());
            } else {
                const usize nm1 = n - 1;
                Array<f64> h(nm1, salloc);
                Array<f64> delta(nm1, salloc);
                for (usize i = 0; i < nm1; ++i) {
                    h[i] = x[i + 1] - x[i];
                    delta[i] = (y[i + 1] - y[i]) / h[i];
                }
                Array<f64> d(n, 0.0, salloc);
                compute_makima_slopes(delta.data(), n, d.data(), salloc);
                eval_hermite(x.data(), y.data(), h.data(), d.data(), n, xq.data(), nq, out.data());
            }
            break;
        case Interp1Method::Cubic:
            if (n < 3) {
                eval_linear(x.data(), y.data(), n, xq.data(), nq, out.data());
            } else if (!is_uniform_grid(x)) {
                eval_spline(x.data(), y.data(), n, xq.data(), nq, out.data(), salloc);
            } else {
                const f64 step = x[1] - x[0];
                Array<f64> ypad(n + 2, salloc);
                for (usize j = 0; j < n; ++j) ypad[j + 1] = y[j];
                ypad[0] = 3.0 * y[0] - 3.0 * y[1] + y[2];
                ypad[n + 1] = 3.0 * y[n - 1] - 3.0 * y[n - 2] + y[n - 3];
                for (usize k = 0; k < nq; ++k) {
                    if (is_nan(xq[k])) {
                        out[k] = nan;
                        continue;
                    }
                    const usize i = find_interval(x.data(), n, xq[k]);
                    const f64 t = (xq[k] - x[i]) / step;
                    out[k] = keys_cubic(1.0 + t) * ypad[i] + keys_cubic(t) * ypad[i + 1] +
                             keys_cubic(1.0 - t) * ypad[i + 2] + keys_cubic(2.0 - t) * ypad[i + 3];
                }
            }
            break;
    }

    // Apply out-of-range extrapolation policy.
    const f64 lo = x[0];
    const f64 hi = x[n - 1];
    const bool method_extraps = (method == Interp1Method::Spline || method == Interp1Method::Pchip ||
                                 method == Interp1Method::Makima);
    for (usize k = 0; k < nq; ++k) {
        const f64 q = xq[k];
        if (is_nan(q) || (q >= lo && q <= hi)) continue;
        switch (extrap) {
            case ExtrapMode::Constant:
                out[k] = extrap_val;
                break;
            case ExtrapMode::Method:
                if (method == Interp1Method::Previous) {
                    out[k] = (q > hi) ? y[n - 1] : nan;
                } else if (method == Interp1Method::Next) {
                    out[k] = (q < lo) ? y[0] : nan;
                }
                break;
            case ExtrapMode::Default:
                if (!method_extraps) out[k] = nan;
                break;
        }
    }
    return true;
}

// ============================================================================
// 3. Piecewise Polynomial Builders & Evaluation (PP-Form)
// ============================================================================

bool spline_pp_coefs(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> coefs_colmajor,
                     Allocator* scratch_alloc) {
    const usize n = x.size();
    if (n < 2 || y.size() != n) return false;
    const usize nm1 = n - 1;
    if (coefs_colmajor.size() < nm1 * 4) return false;

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();
    Array<f64> h(nm1, salloc);
    for (usize i = 0; i < nm1; ++i) h[i] = x[i + 1] - x[i];
    Array<f64> sigma(n, salloc);
    compute_spline_sigma(y.data(), n, h.data(), sigma.data(), salloc);

    for (usize i = 0; i < nm1; ++i) {
        const f64 hi = h[i];
        coefs_colmajor[i + 0 * nm1] = (sigma[i + 1] - sigma[i]) / (6.0 * hi);
        coefs_colmajor[i + 1 * nm1] = 0.5 * sigma[i];
        coefs_colmajor[i + 2 * nm1] =
            (y[i + 1] - y[i]) / hi - hi * (2.0 * sigma[i] + sigma[i + 1]) / 6.0;
        coefs_colmajor[i + 3 * nm1] = y[i];
    }
    return true;
}

bool pchip_pp_coefs(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> coefs_colmajor,
                    Allocator* scratch_alloc) {
    const usize n = x.size();
    if (n < 2 || y.size() != n) return false;
    const usize nm1 = n - 1;
    if (coefs_colmajor.size() < nm1 * 4) return false;

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();
    Array<f64> h(nm1, salloc);
    Array<f64> delta(nm1, salloc);
    for (usize i = 0; i < nm1; ++i) {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }
    Array<f64> d(n, 0.0, salloc);
    compute_pchip_slopes(h.data(), delta.data(), n, d.data());

    for (usize i = 0; i < nm1; ++i) {
        const f64 hi = h[i];
        coefs_colmajor[i + 0 * nm1] = (d[i] + d[i + 1] - 2.0 * delta[i]) / (hi * hi);
        coefs_colmajor[i + 1 * nm1] = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi;
        coefs_colmajor[i + 2 * nm1] = d[i];
        coefs_colmajor[i + 3 * nm1] = y[i];
    }
    return true;
}

bool makima_pp_coefs(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> coefs_colmajor,
                     Allocator* scratch_alloc) {
    const usize n = x.size();
    if (n < 2 || y.size() != n) return false;
    const usize nm1 = n - 1;
    if (coefs_colmajor.size() < nm1 * 4) return false;

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();
    Array<f64> h(nm1, salloc);
    Array<f64> delta(nm1, salloc);
    for (usize i = 0; i < nm1; ++i) {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }
    Array<f64> d(n, 0.0, salloc);
    compute_makima_slopes(delta.data(), n, d.data(), salloc);

    for (usize i = 0; i < nm1; ++i) {
        const f64 hi = h[i];
        coefs_colmajor[i + 0 * nm1] = (d[i] + d[i + 1] - 2.0 * delta[i]) / (hi * hi);
        coefs_colmajor[i + 1 * nm1] = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi;
        coefs_colmajor[i + 2 * nm1] = d[i];
        coefs_colmajor[i + 3 * nm1] = y[i];
    }
    return true;
}

bool ppval(ArrayRef<f64> breaks, ArrayRef<f64> coefs_colmajor, usize pieces, usize order,
           ArrayRef<f64> xq, MutArrayRef<f64> out) noexcept {
    if (pieces == 0 || order == 0 || breaks.size() < pieces + 1 ||
        coefs_colmajor.size() < pieces * order || out.size() < xq.size()) {
        return false;
    }
    const usize nq = xq.size();
    for (usize i = 0; i < nq; ++i) {
        const f64 xi = xq[i];
        if (is_nan(xi)) {
            out[i] = Limits<f64>::quiet_nan();
            continue;
        }
        usize j = 0;
        if (xi <= breaks[0]) {
            j = 0;
        } else if (xi >= breaks[pieces]) {
            j = pieces - 1;
        } else {
            // Binary search for largest j in [0, pieces-1] with breaks[j] <= xi.
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
        f64 y = coefs_colmajor[j];
        for (usize k = 1; k < order; ++k) {
            y = y * u + coefs_colmajor[k * pieces + j];
        }
        out[i] = y;
    }
    return true;
}

// ============================================================================
// 4. 2D & 3D Gridded Interpolation
// ============================================================================

bool interp2_pointwise(ArrayRef<f64> v_colmajor, usize rows, usize cols, ArrayRef<f64> x_grid,
                       ArrayRef<f64> y_grid, ArrayRef<f64> xq, ArrayRef<f64> yq,
                       MutArrayRef<f64> out, Interp2Method method, Allocator* scratch_alloc) {
    if (x_grid.size() != cols || y_grid.size() != rows || v_colmajor.size() < rows * cols) {
        return false;
    }
    if (xq.size() != yq.size() || out.size() < xq.size()) return false;
    if (!is_strictly_increasing(x_grid) || !is_strictly_increasing(y_grid)) return false;

    const usize nq = xq.size();
    if (nq == 0) return true;

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();

    if (method == Interp2Method::Cubic) {
        if (rows < 3 || cols < 3 || !is_uniform_grid(x_grid) || !is_uniform_grid(y_grid)) {
            return false;
        }
        Array<f64> vpad(salloc);
        build_vpad_2d(v_colmajor.data(), rows, cols, vpad);
        for (usize k = 0; k < nq; ++k) {
            out[k] = sample_2d_basic(v_colmajor.data(), vpad.data(), rows, cols, x_grid.data(),
                                     y_grid.data(), xq[k], yq[k], method);
        }
        return true;
    }

    if (method == Interp2Method::Spline) {
        // Transpose V to row-major once; precompute x-direction spline sigma per row.
        Array<f64> vrow(rows * cols, salloc);
        for (usize i = 0; i < rows; ++i) {
            for (usize j = 0; j < cols; ++j) vrow[i * cols + j] = v_colmajor[j * rows + i];
        }
        Array<f64> col_tmp(rows, salloc);
        if (cols >= 3) {
            const usize cm1 = cols - 1;
            Array<f64> hx(cm1, salloc);
            for (usize j = 0; j < cm1; ++j) hx[j] = x_grid[j + 1] - x_grid[j];
            Array<f64> row_sigmas(rows * cols, salloc);
            for (usize i = 0; i < rows; ++i) {
                compute_spline_sigma(vrow.data() + i * cols, cols, hx.data(),
                                     row_sigmas.data() + i * cols, salloc);
            }
            for (usize k = 0; k < nq; ++k) {
                if (is_nan(xq[k]) || is_nan(yq[k])) {
                    out[k] = Limits<f64>::quiet_nan();
                    continue;
                }
                const usize ix = find_interval(x_grid.data(), cols, xq[k]);
                const f64 dx = xq[k] - x_grid[ix];
                const f64 dx1 = x_grid[ix + 1] - xq[k];
                const f64 hi = hx[ix];
                const f64 c0 = dx1 * dx1 * dx1 / (6.0 * hi) - hi * dx1 / 6.0;
                const f64 c1 = dx * dx * dx / (6.0 * hi) - hi * dx / 6.0;
                const f64 w0 = dx1 / hi;
                const f64 w1 = dx / hi;
                for (usize i = 0; i < rows; ++i) {
                    const f64* rptr = vrow.data() + i * cols;
                    const f64* sptr = row_sigmas.data() + i * cols;
                    col_tmp[i] = sptr[ix] * c0 + sptr[ix + 1] * c1 + rptr[ix] * w0 + rptr[ix + 1] * w1;
                }
                eval_spline(y_grid.data(), col_tmp.data(), rows, &yq[k], 1, &out[k], salloc);
            }
        } else {
            for (usize k = 0; k < nq; ++k) {
                for (usize i = 0; i < rows; ++i) {
                    eval_linear(x_grid.data(), vrow.data() + i * cols, cols, &xq[k], 1, &col_tmp[i]);
                }
                eval_spline(y_grid.data(), col_tmp.data(), rows, &yq[k], 1, &out[k], salloc);
            }
        }
        return true;
    }

    for (usize k = 0; k < nq; ++k) {
        out[k] = sample_2d_basic(v_colmajor.data(), nullptr, rows, cols, x_grid.data(),
                                 y_grid.data(), xq[k], yq[k], method);
    }
    return true;
}

bool interp2_grid(ArrayRef<f64> v_colmajor, usize rows, usize cols, ArrayRef<f64> x_grid,
                  ArrayRef<f64> y_grid, ArrayRef<f64> xq_cols, ArrayRef<f64> yq_rows,
                  MutArrayRef<f64> out_colmajor, Interp2Method method, Allocator* scratch_alloc) {
    if (x_grid.size() != cols || y_grid.size() != rows || v_colmajor.size() < rows * cols) {
        return false;
    }
    const usize nx = xq_cols.size();
    const usize ny = yq_rows.size();
    if (out_colmajor.size() < nx * ny) return false;
    if (!is_strictly_increasing(x_grid) || !is_strictly_increasing(y_grid)) return false;
    if (nx == 0 || ny == 0) return true;

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();

    if (method == Interp2Method::Cubic) {
        if (rows < 3 || cols < 3 || !is_uniform_grid(x_grid) || !is_uniform_grid(y_grid)) {
            return false;
        }
        Array<f64> vpad(salloc);
        build_vpad_2d(v_colmajor.data(), rows, cols, vpad);
        for (usize j = 0; j < nx; ++j) {
            for (usize i = 0; i < ny; ++i) {
                out_colmajor[j * ny + i] =
                    sample_2d_basic(v_colmajor.data(), vpad.data(), rows, cols, x_grid.data(),
                                    y_grid.data(), xq_cols[j], yq_rows[i], method);
            }
        }
        return true;
    }

    if (method == Interp2Method::Spline) {
        // Separable tensor-product spline: interpolate each row along x_grid -> (rows x nx),
        // then each column along y_grid -> (ny x nx).
        Array<f64> vrow(cols, salloc);
        Array<f64> mid_colmajor(rows * nx, salloc);
        Array<f64> row_out(nx, salloc);
        for (usize i = 0; i < rows; ++i) {
            for (usize j = 0; j < cols; ++j) vrow[j] = v_colmajor[j * rows + i];
            eval_spline(x_grid.data(), vrow.data(), cols, xq_cols.data(), nx, row_out.data(), salloc);
            for (usize j = 0; j < nx; ++j) mid_colmajor[j * rows + i] = row_out[j];
        }
        for (usize j = 0; j < nx; ++j) {
            eval_spline(y_grid.data(), mid_colmajor.data() + j * rows, rows, yq_rows.data(), ny,
                        out_colmajor.data() + j * ny, salloc);
        }
        return true;
    }

    for (usize j = 0; j < nx; ++j) {
        for (usize i = 0; i < ny; ++i) {
            out_colmajor[j * ny + i] =
                sample_2d_basic(v_colmajor.data(), nullptr, rows, cols, x_grid.data(),
                                y_grid.data(), xq_cols[j], yq_rows[i], method);
        }
    }
    return true;
}

bool interp3_pointwise(ArrayRef<f64> v_colmajor, usize rows, usize cols, usize pages,
                       ArrayRef<f64> x_grid, ArrayRef<f64> y_grid, ArrayRef<f64> z_grid,
                       ArrayRef<f64> xq, ArrayRef<f64> yq, ArrayRef<f64> zq, MutArrayRef<f64> out,
                       Interp2Method method) noexcept {
    if (method != Interp2Method::Linear && method != Interp2Method::Nearest) return false;
    if (x_grid.size() != cols || y_grid.size() != rows || z_grid.size() != pages ||
        v_colmajor.size() < rows * cols * pages) {
        return false;
    }
    const usize nq = xq.size();
    if (yq.size() != nq || zq.size() != nq || out.size() < nq) return false;
    if (!is_strictly_increasing(x_grid) || !is_strictly_increasing(y_grid) ||
        !is_strictly_increasing(z_grid)) {
        return false;
    }

    auto val = [&](usize i, usize j, usize k) noexcept {
        return v_colmajor[k * rows * cols + j * rows + i];
    };

    const f64 nan = Limits<f64>::quiet_nan();
    for (usize q = 0; q < nq; ++q) {
        const usize ix = find_cell(x_grid.data(), cols, xq[q]);
        const usize iy = find_cell(y_grid.data(), rows, yq[q]);
        const usize iz = find_cell(z_grid.data(), pages, zq[q]);
        if (ix == kInvalidIndex || iy == kInvalidIndex || iz == kInvalidIndex) {
            out[q] = nan;
            continue;
        }
        if (method == Interp2Method::Nearest) {
            const usize cx = (xq[q] - x_grid[ix] <= x_grid[ix + 1] - xq[q]) ? ix : (ix + 1);
            const usize cy = (yq[q] - y_grid[iy] <= y_grid[iy + 1] - yq[q]) ? iy : (iy + 1);
            const usize cz = (zq[q] - z_grid[iz] <= z_grid[iz + 1] - zq[q]) ? iz : (iz + 1);
            out[q] = val(cy, cx, cz);
            continue;
        }
        const f64 tx = (xq[q] - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix]);
        const f64 ty = (yq[q] - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy]);
        const f64 tz = (zq[q] - z_grid[iz]) / (z_grid[iz + 1] - z_grid[iz]);
        const f64 c00 = (1.0 - tx) * val(iy, ix, iz) + tx * val(iy, ix + 1, iz);
        const f64 c10 = (1.0 - tx) * val(iy + 1, ix, iz) + tx * val(iy + 1, ix + 1, iz);
        const f64 c01 = (1.0 - tx) * val(iy, ix, iz + 1) + tx * val(iy, ix + 1, iz + 1);
        const f64 c11 = (1.0 - tx) * val(iy + 1, ix, iz + 1) + tx * val(iy + 1, ix + 1, iz + 1);
        const f64 c0 = (1.0 - ty) * c00 + ty * c10;
        const f64 c1 = (1.0 - ty) * c01 + ty * c11;
        out[q] = (1.0 - tz) * c0 + tz * c1;
    }
    return true;
}

} // namespace keel::interp
