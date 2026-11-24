#include "keel/interp.h"
#include "keel/math.h"

#include <cstdio>

namespace {

keel::i32 g_failed = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [line %d]: %s\n", keel::as<int>(line), expr);
        ++g_failed;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* expr, keel::i32 line) {
    if (keel::is_nan(got) || keel::abs(got - expected) > tol) {
        std::fprintf(stderr, "FAIL [line %d]: %s (got %.15g, expected %.15g)\n",
                     keel::as<int>(line), expr, got, expected);
        ++g_failed;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_NEAR(got, expected, tol) check_near((got), (expected), (tol), #got " ~= " #expected, __LINE__)

void test_interp1_and_pp() {
    const keel::f64 x[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    const keel::f64 y[5] = {0.0, 1.0, 4.0, 9.0, 16.0}; // y = x^2
    const keel::f64 xq[4] = {-1.0, 0.5, 2.5, 5.0};
    keel::f64 out[4] = {};

    // Linear default extrap -> NaN outside, linear inside.
    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Linear));
    CHECK(keel::is_nan(out[0]));
    CHECK_NEAR(out[1], 0.5, 1e-14);
    CHECK_NEAR(out[2], 6.5, 1e-14);
    CHECK(keel::is_nan(out[3]));

    // Linear constant extrap.
    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Linear,
                        keel::interp::ExtrapMode::Constant, -99.0));
    CHECK_NEAR(out[0], -99.0, 1e-14);
    CHECK_NEAR(out[3], -99.0, 1e-14);

    // Nearest tie-break rounds up to higher index.
    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Nearest));
    CHECK_NEAR(out[1], 1.0, 1e-14);
    CHECK_NEAR(out[2], 9.0, 1e-14);

    // Previous / Next.
    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Previous,
                        keel::interp::ExtrapMode::Method));
    CHECK(keel::is_nan(out[0]));
    CHECK_NEAR(out[1], 0.0, 1e-14);
    CHECK_NEAR(out[3], 16.0, 1e-14);

    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Next,
                        keel::interp::ExtrapMode::Method));
    CHECK_NEAR(out[0], 0.0, 1e-14);
    CHECK_NEAR(out[1], 1.0, 1e-14);
    CHECK(keel::is_nan(out[3]));

    // Not-a-knot cubic spline reproduces x^2 (and cubics!) EXACTLY, including extrapolation!
    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Spline));
    CHECK_NEAR(out[0], 1.0, 1e-12);
    CHECK_NEAR(out[1], 0.25, 1e-13);
    CHECK_NEAR(out[2], 6.25, 1e-13);
    CHECK_NEAR(out[3], 25.0, 1e-12);

    // 3-point not-a-knot spline reproduces parabola x^2 exactly.
    keel::f64 out3[1] = {};
    const keel::f64 q3[1] = {1.5};
    CHECK(keel::interp::interp1({x, 3}, {y, 3}, {q3, 1}, {out3, 1}, keel::interp::Interp1Method::Spline));
    CHECK_NEAR(out3[0], 2.25, 1e-13);

    // PP-form parity for spline, pchip, makima.
    keel::f64 coefs[4 * 4] = {};
    keel::f64 pp_out[4] = {};
    CHECK(keel::interp::spline_pp_coefs({x, 5}, {y, 5}, {coefs, 16}));
    CHECK(keel::interp::ppval({x, 5}, {coefs, 16}, 4, 4, {xq, 4}, {pp_out, 4}));
    for (keel::usize i = 0; i < 4; ++i) CHECK_NEAR(pp_out[i], out[i], 1e-12);

    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Pchip));
    CHECK(keel::interp::pchip_pp_coefs({x, 5}, {y, 5}, {coefs, 16}));
    CHECK(keel::interp::ppval({x, 5}, {coefs, 16}, 4, 4, {xq, 4}, {pp_out, 4}));
    for (keel::usize i = 0; i < 4; ++i) CHECK_NEAR(pp_out[i], out[i], 1e-12);

    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Makima));
    CHECK(keel::interp::makima_pp_coefs({x, 5}, {y, 5}, {coefs, 16}));
    CHECK(keel::interp::ppval({x, 5}, {coefs, 16}, 4, 4, {xq, 4}, {pp_out, 4}));
    for (keel::usize i = 0; i < 4; ++i) CHECK_NEAR(pp_out[i], out[i], 1e-12);

    // Keys cubic on uniform grid.
    CHECK(keel::interp::interp1({x, 5}, {y, 5}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Cubic));
    CHECK(keel::is_nan(out[0]));
    CHECK_NEAR(out[1], 0.25, 1e-12);
    CHECK_NEAR(out[2], 6.25, 1e-12);
}

void test_interp2_and_interp3() {
    // 4x4 grid of f(x, y) = x^2 + 2*y^2.
    const keel::f64 xg[4] = {0.0, 1.0, 2.0, 3.0};
    const keel::f64 yg[4] = {0.0, 1.0, 2.0, 3.0};
    keel::f64 v2[16] = {};
    for (keel::usize j = 0; j < 4; ++j) {
        for (keel::usize i = 0; i < 4; ++i) {
            v2[j * 4 + i] = xg[j] * xg[j] + 2.0 * yg[i] * yg[i];
        }
    }

    const keel::f64 qx[2] = {1.5, 2.25};
    const keel::f64 qy[2] = {0.5, 1.75};
    keel::f64 out2[2] = {};

    // Bicubic & Spline reproduce quadratic x^2 + 2y^2 accurately.
    CHECK(keel::interp::interp2_pointwise({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {qx, 2}, {qy, 2}, {out2, 2},
                                  keel::interp::Interp2Method::Spline));
    CHECK_NEAR(out2[0], 1.5 * 1.5 + 2.0 * 0.5 * 0.5, 1e-12);
    CHECK_NEAR(out2[1], 2.25 * 2.25 + 2.0 * 1.75 * 1.75, 1e-12);

    CHECK(keel::interp::interp2_pointwise({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {qx, 2}, {qy, 2}, {out2, 2},
                                  keel::interp::Interp2Method::Cubic));
    CHECK_NEAR(out2[0], 1.5 * 1.5 + 2.0 * 0.5 * 0.5, 1e-12);
    CHECK_NEAR(out2[1], 2.25 * 2.25 + 2.0 * 1.75 * 1.75, 1e-12);

    // Grid expansion (2x2 output).
    keel::f64 grid_out[4] = {};
    CHECK(keel::interp::interp2_grid({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {qx, 2}, {qy, 2}, {grid_out, 4},
                             keel::interp::Interp2Method::Spline));
    CHECK_NEAR(grid_out[0], 1.5 * 1.5 + 2.0 * 0.5 * 0.5, 1e-12);   // (row 0, col 0)
    CHECK_NEAR(grid_out[1], 1.5 * 1.5 + 2.0 * 1.75 * 1.75, 1e-12); // (row 1, col 0)
    CHECK_NEAR(grid_out[2], 2.25 * 2.25 + 2.0 * 0.5 * 0.5, 1e-12); // (row 0, col 1)
    CHECK_NEAR(grid_out[3], 2.25 * 2.25 + 2.0 * 1.75 * 1.75, 1e-12);

    // 3D trilinear on 2x2x2 cube f(x,y,z) = x + 2y + 3z.
    const keel::f64 g2[2] = {0.0, 1.0};
    keel::f64 v3[8] = {};
    for (keel::usize k = 0; k < 2; ++k) {
        for (keel::usize j = 0; j < 2; ++j) {
            for (keel::usize i = 0; i < 2; ++i) {
                v3[k * 4 + j * 2 + i] = g2[j] + 2.0 * g2[i] + 3.0 * g2[k];
            }
        }
    }
    const keel::f64 q3x[1] = {0.25};
    const keel::f64 q3y[1] = {0.5};
    const keel::f64 q3z[1] = {0.75};
    keel::f64 out3[1] = {};
    CHECK(keel::interp::interp3_pointwise({v3, 8}, 2, 2, 2, {g2, 2}, {g2, 2}, {g2, 2}, {q3x, 1}, {q3y, 1},
                                  {q3z, 1}, {out3, 1}, keel::interp::Interp2Method::Linear));
    CHECK_NEAR(out3[0], 0.25 + 2.0 * 0.5 + 3.0 * 0.75, 1e-14);
}

void test_spline_pp_calculus() {
    // aveknt
    const keel::f64 t[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    keel::f64 avg[3] = {};
    CHECK(keel::interp::aveknt({t, 6}, 3, {avg, 3}));
    CHECK_NEAR(avg[0], 0.0, 1e-15);
    CHECK_NEAR(avg[1], 0.5, 1e-15);
    CHECK_NEAR(avg[2], 1.0, 1e-15);

    // augknt
    const keel::f64 br[3] = {0.0, 0.5, 1.0};
    keel::f64 aug[7] = {};
    CHECK(keel::interp::augknt({br, 3}, 3, {aug, 7}));
    CHECK_NEAR(aug[0], 0.0, 1e-15);
    CHECK_NEAR(aug[2], 0.0, 1e-15);
    CHECK_NEAR(aug[3], 0.5, 1e-15);
    CHECK_NEAR(aug[4], 1.0, 1e-15);
    CHECK_NEAR(aug[6], 1.0, 1e-15);

    // brk2knt & knt2brk round-trip
    const keel::usize mults[3] = {3, 1, 3};
    keel::Array<keel::f64> knots;
    CHECK(keel::interp::brk2knt({br, 3}, {mults, 3}, knots));
    CHECK(knots.size() == 7);
    keel::Array<keel::f64> br_out;
    keel::Array<keel::usize> m_out;
    keel::interp::knt2brk(knots.as_ref(), br_out, m_out);
    CHECK(br_out.size() == 3);
    CHECK(m_out[0] == 3 && m_out[1] == 1 && m_out[2] == 3);

    // pp_deriv and pp_integ on f(x) = x^2 over [0, 1, 2]
    const keel::f64 x[3] = {0.0, 1.0, 2.0};
    const keel::f64 y[3] = {0.0, 1.0, 4.0};
    keel::f64 coefs[2 * 4] = {};
    CHECK(keel::interp::spline_pp_coefs({x, 3}, {y, 3}, {coefs, 8}));

    keel::Array<keel::f64> dcoefs;
    keel::usize dord = 0;
    CHECK(keel::interp::pp_deriv({coefs, 8}, 2, 4, 1, 1, dcoefs, dord));
    CHECK(dord == 3);
    const keel::f64 xq[2] = {0.5, 1.5};
    keel::f64 dy[2] = {};
    CHECK(keel::interp::ppval_nd({x, 3}, dcoefs.as_ref(), 2, dord, 1, {xq, 2}, {dy, 2}));
    CHECK_NEAR(dy[0], 1.0, 1e-12); // d/dx(x^2) at 0.5 = 1.0
    CHECK_NEAR(dy[1], 3.0, 1e-12); // d/dx(x^2) at 1.5 = 3.0

    keel::f64 icoefs[2 * 5] = {};
    CHECK(keel::interp::pp_integ({x, 3}, {coefs, 8}, 2, 4, 1, {icoefs, 10}));
    keel::f64 iy[2] = {};
    CHECK(keel::interp::ppval_nd({x, 3}, {icoefs, 10}, 2, 5, 1, {xq, 2}, {iy, 2}));
    CHECK_NEAR(iy[0], 0.5 * 0.5 * 0.5 / 3.0, 1e-12);
    CHECK_NEAR(iy[1], 1.5 * 1.5 * 1.5 / 3.0, 1e-12);
}

} // namespace

int main() {
    test_interp1_and_pp();
    test_interp2_and_interp3();
    test_spline_pp_calculus();

    if (g_failed != 0) {
        std::fprintf(stderr, "test_interp: %d failure(s)\n", keel::as<int>(g_failed));
        return 1;
    }
    std::printf("test_interp: all tests passed\n");
    return 0;
}
