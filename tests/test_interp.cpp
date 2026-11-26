#include "keel/interp.h"
#include "keel/format.h"
#include "keel/math.h"
#include "keel/string.h"

namespace {

keel::i32 g_failed = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        keel::eprintln("FAIL [line %d]: %s", line, expr);
        ++g_failed;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* expr, keel::i32 line) {
    if (keel::is_nan(got) || keel::abs(got - expected) > tol) {
        keel::eprintln("FAIL [line %d]: %s (got %.15g, expected %.15g)", line, expr, got, expected);
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

    // PCHIP flat boundary intervals must NOT overshoot!
    const keel::f64 y_flat[5] = {1.0, 1.0, 5.0, 6.0, 6.0};
    const keel::f64 q_flat[4] = {0.25, 0.75, 3.25, 3.75};
    keel::f64 out_flat[4] = {};
    CHECK(keel::interp::interp1({x, 5}, {y_flat, 5}, {q_flat, 4}, {out_flat, 4}, keel::interp::Interp1Method::Pchip));
    CHECK_NEAR(out_flat[0], 1.0, 1e-14);
    CHECK_NEAR(out_flat[1], 1.0, 1e-14);
    CHECK_NEAR(out_flat[2], 6.0, 1e-14);
    CHECK_NEAR(out_flat[3], 6.0, 1e-14);

    // Duplicate / non-increasing / NaN knots rejected.
    const keel::f64 x_dup[4] = {0.0, 1.0, 1.0, 2.0};
    const keel::f64 y4[4] = {0.0, 1.0, 2.0, 3.0};
    CHECK(!keel::interp::interp1({x_dup, 4}, {y4, 4}, {xq, 4}, {out, 4}, keel::interp::Interp1Method::Spline));
    const keel::f64 x_nan[3] = {0.0, keel::kNaN, 2.0};
    CHECK(!keel::interp::spline_pp_coefs({x_nan, 3}, {y4, 3}, {out, 4}));

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

    // Reject degenerate 1-row/1-col grids cleanly (must not underflow find_interval).
    const keel::f64 g1[1] = {0.0};
    CHECK(!keel::interp::interp2_pointwise({v2, 4}, 1, 4, {xg, 4}, {g1, 1}, {qx, 2}, {qy, 2}, {out2, 2},
                                           keel::interp::Interp2Method::Spline));

    CHECK(keel::interp::interp2_pointwise({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {qx, 2}, {qy, 2}, {out2, 2},
                                          keel::interp::Interp2Method::Spline));
    CHECK_NEAR(out2[0], 1.5 * 1.5 + 2.0 * 0.5 * 0.5, 1e-12);
    CHECK_NEAR(out2[1], 2.25 * 2.25 + 2.0 * 1.75 * 1.75, 1e-12);

    CHECK(keel::interp::interp2_pointwise({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {qx, 2}, {qy, 2}, {out2, 2},
                                          keel::interp::Interp2Method::Cubic));
    CHECK_NEAR(out2[0], 1.5 * 1.5 + 2.0 * 0.5 * 0.5, 1e-12);
    CHECK_NEAR(out2[1], 2.25 * 2.25 + 2.0 * 1.75 * 1.75, 1e-12);

    keel::f64 grid_out[4] = {};
    CHECK(keel::interp::interp2_grid({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {qx, 2}, {qy, 2}, {grid_out, 4},
                                     keel::interp::Interp2Method::Spline));
    CHECK_NEAR(grid_out[0], 1.5 * 1.5 + 2.0 * 0.5 * 0.5, 1e-12);
    CHECK_NEAR(grid_out[1], 1.5 * 1.5 + 2.0 * 1.75 * 1.75, 1e-12);
    CHECK_NEAR(grid_out[2], 2.25 * 2.25 + 2.0 * 0.5 * 0.5, 1e-12);
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
    CHECK(keel::interp::interp3_pointwise({v3, 8}, 2, 2, 2, {g2, 2}, {g2, 2}, {g2, 2}, {q3x, 1}, {q3y, 1}, {q3z, 1},
                                          {out3, 1}, keel::interp::Interp2Method::Linear));
    CHECK_NEAR(out3[0], 0.25 + 2.0 * 0.5 + 3.0 * 0.75, 1e-14);
}

void test_spline_pp_calculus() {
    const keel::f64 t[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    keel::f64 avg[3] = {};
    CHECK(keel::interp::aveknt({t, 6}, 3, {avg, 3}));
    CHECK_NEAR(avg[0], 0.0, 1e-15);
    CHECK_NEAR(avg[1], 0.5, 1e-15);
    CHECK_NEAR(avg[2], 1.0, 1e-15);

    const keel::f64 br[3] = {0.0, 0.5, 1.0};
    keel::f64 aug[7] = {};
    CHECK(keel::interp::augknt({br, 3}, 3, {aug, 7}));
    CHECK_NEAR(aug[0], 0.0, 1e-15);
    CHECK_NEAR(aug[2], 0.0, 1e-15);
    CHECK_NEAR(aug[3], 0.5, 1e-15);
    CHECK_NEAR(aug[4], 1.0, 1e-15);
    CHECK_NEAR(aug[6], 1.0, 1e-15);

    // Single-knot augknt must be rejected cleanly (no buffer overflow!).
    const keel::f64 br1[1] = {0.0};
    keel::f64 aug1[5] = {};
    CHECK(!keel::interp::augknt({br1, 1}, 3, {aug1, 5}));

    const keel::usize mults[3] = {3, 1, 3};
    keel::Array<keel::f64> knots;
    CHECK(keel::interp::brk2knt({br, 3}, {mults, 3}, knots));
    CHECK(knots.size() == 7);
    keel::Array<keel::f64> br_out;
    keel::Array<keel::usize> m_out;
    keel::interp::knt2brk(knots.as_ref(), br_out, m_out);
    CHECK(br_out.size() == 3);
    CHECK(m_out[0] == 3 && m_out[1] == 1 && m_out[2] == 3);

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
    CHECK_NEAR(dy[0], 1.0, 1e-12);
    CHECK_NEAR(dy[1], 3.0, 1e-12);

    keel::f64 icoefs[2 * 5] = {};
    CHECK(keel::interp::pp_integ({x, 3}, {coefs, 8}, 2, 4, 1, {icoefs, 10}));
    keel::f64 iy[2] = {};
    CHECK(keel::interp::ppval_nd({x, 3}, {icoefs, 10}, 2, 5, 1, {xq, 2}, {iy, 2}));
    CHECK_NEAR(iy[0], 0.5 * 0.5 * 0.5 / 3.0, 1e-12);
    CHECK_NEAR(iy[1], 1.5 * 1.5 * 1.5 / 3.0, 1e-12);
}

void dump_f64_array(const char* name, keel::ArrayRef<keel::f64> v) {
    keel::print("%s:", name);
    for (keel::usize i = 0; i < v.size(); ++i) {
        keel::print(" %.17g", v[i]);
    }
    keel::println("");
}

void dump_reference() {
    // 1. Non-uniform 1D interpolation: spline, pchip, makima, linear, nearest
    const keel::f64 x_nu[5] = {0.0, 0.5, 2.0, 3.0, 5.0};
    const keel::f64 y_nu[5] = {0.0, 1.2, -0.5, 2.3, 1.0};
    const keel::f64 xq_nu[6] = {-0.5, 0.25, 1.1, 2.7, 4.2, 5.5};
    keel::f64 out6[6] = {};

    (void)keel::interp::interp1({x_nu, 5}, {y_nu, 5}, {xq_nu, 6}, {out6, 6}, keel::interp::Interp1Method::Spline);
    dump_f64_array("spline_nu", out6);
    (void)keel::interp::interp1({x_nu, 5}, {y_nu, 5}, {xq_nu, 6}, {out6, 6}, keel::interp::Interp1Method::Pchip);
    dump_f64_array("pchip_nu", out6);
    (void)keel::interp::interp1({x_nu, 5}, {y_nu, 5}, {xq_nu, 6}, {out6, 6}, keel::interp::Interp1Method::Makima);
    dump_f64_array("makima_nu", out6);

    const keel::f64 xq_in[4] = {0.25, 1.1, 2.7, 4.2};
    keel::f64 out4[4] = {};
    (void)keel::interp::interp1({x_nu, 5}, {y_nu, 5}, {xq_in, 4}, {out4, 4}, keel::interp::Interp1Method::Linear);
    dump_f64_array("linear_nu", out4);

    // 2. Flat-boundary PCHIP & Makima
    const keel::f64 x_f[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    const keel::f64 y_f[5] = {1.0, 1.0, 5.0, 6.0, 6.0};
    const keel::f64 xq_f[6] = {0.25, 0.5, 0.75, 3.25, 3.5, 3.75};
    (void)keel::interp::interp1({x_f, 5}, {y_f, 5}, {xq_f, 6}, {out6, 6}, keel::interp::Interp1Method::Pchip);
    dump_f64_array("pchip_flat", out6);
    (void)keel::interp::interp1({x_f, 5}, {y_f, 5}, {xq_f, 6}, {out6, 6}, keel::interp::Interp1Method::Makima);
    dump_f64_array("makima_flat", out6);

    // 3. 2D gridded interpolation (non-uniform Linear + Spline, uniform Cubic)
    const keel::f64 xg[4] = {0.0, 0.7, 1.8, 3.0};
    const keel::f64 yg[4] = {-1.0, 0.2, 1.5, 2.5};
    keel::f64 v2[16] = {};
    for (keel::usize j = 0; j < 4; ++j) {
        for (keel::usize i = 0; i < 4; ++i) {
            v2[j * 4 + i] = keel::sin(xg[j]) * keel::cos(yg[i]) + 0.3 * xg[j] * yg[i];
        }
    }
    const keel::f64 q2x[3] = {0.35, 1.2, 2.4};
    const keel::f64 q2y[3] = {-0.4, 0.8, 2.1};
    keel::f64 out2[3] = {};
    (void)keel::interp::interp2_pointwise({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {q2x, 3}, {q2y, 3}, {out2, 3},
                                          keel::interp::Interp2Method::Linear);
    dump_f64_array("interp2_linear", out2);
    (void)keel::interp::interp2_pointwise({v2, 16}, 4, 4, {xg, 4}, {yg, 4}, {q2x, 3}, {q2y, 3}, {out2, 3},
                                          keel::interp::Interp2Method::Spline);
    dump_f64_array("interp2_spline", out2);

    // Uniform 2D cubic
    const keel::f64 ug[4] = {0.0, 1.0, 2.0, 3.0};
    keel::f64 v2u[16] = {};
    for (keel::usize j = 0; j < 4; ++j) {
        for (keel::usize i = 0; i < 4; ++i) {
            v2u[j * 4 + i] = keel::sin(ug[j]) + keel::cos(ug[i]);
        }
    }
    const keel::f64 qux[3] = {0.4, 1.3, 2.6};
    const keel::f64 quy[3] = {0.7, 1.8, 2.2};
    (void)keel::interp::interp2_pointwise({v2u, 16}, 4, 4, {ug, 4}, {ug, 4}, {qux, 3}, {quy, 3}, {out2, 3},
                                          keel::interp::Interp2Method::Cubic);
    dump_f64_array("interp2_cubic", out2);

    // 4. 3D trilinear
    const keel::f64 g3x[3] = {0.0, 1.0, 3.0};
    const keel::f64 g3y[3] = {-1.0, 0.5, 2.0};
    const keel::f64 g3z[2] = {0.0, 2.0};
    keel::f64 v3[18] = {};
    for (keel::usize k = 0; k < 2; ++k) {
        for (keel::usize j = 0; j < 3; ++j) {
            for (keel::usize i = 0; i < 3; ++i) {
                v3[k * 9 + j * 3 + i] = 2.0 * g3x[j] - g3y[i] + 0.5 * g3z[k] + g3x[j] * g3y[i] * g3z[k];
            }
        }
    }
    const keel::f64 q3x[2] = {0.5, 2.2};
    const keel::f64 q3y[2] = {0.0, 1.25};
    const keel::f64 q3z[2] = {0.8, 1.6};
    keel::f64 out3[2] = {};
    (void)keel::interp::interp3_pointwise({v3, 18}, 3, 3, 2, {g3x, 3}, {g3y, 3}, {g3z, 2}, {q3x, 2}, {q3y, 2}, {q3z, 2},
                                          {out3, 2}, keel::interp::Interp2Method::Linear);
    dump_f64_array("interp3_linear", out3);

    // 5. pp_deriv & pp_integ on non-uniform not-a-knot cubic spline
    keel::f64 sp_coefs[4 * 4] = {};
    (void)keel::interp::spline_pp_coefs({x_nu, 5}, {y_nu, 5}, {sp_coefs, 16});
    keel::Array<keel::f64> dcoefs;
    keel::usize dord = 0;
    (void)keel::interp::pp_deriv({sp_coefs, 16}, 4, 4, 1, 1, dcoefs, dord);
    (void)keel::interp::ppval_nd({x_nu, 5}, dcoefs.as_ref(), 4, dord, 1, {xq_in, 4}, {out4, 4});
    dump_f64_array("pp_deriv1", out4);

    keel::f64 icoefs[4 * 5] = {};
    (void)keel::interp::pp_integ({x_nu, 5}, {sp_coefs, 16}, 4, 4, 1, {icoefs, 20});
    (void)keel::interp::ppval_nd({x_nu, 5}, {icoefs, 20}, 4, 5, 1, {xq_in, 4}, {out4, 4});
    dump_f64_array("pp_integ", out4);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }
    test_interp1_and_pp();
    test_interp2_and_interp3();
    test_spline_pp_calculus();

    if (g_failed != 0) {
        keel::eprintln("test_interp: %d failure(s)", g_failed);
        return 1;
    }
    keel::println("test_interp: all tests passed");
    return 0;
}
