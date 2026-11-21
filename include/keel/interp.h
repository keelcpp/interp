#pragma once

#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::interp {

// ============================================================================
// 1. Grid Validation Helpers
// ============================================================================

/// Returns true iff grid[i] > grid[i-1] for all i = 1..n-1.
[[nodiscard]] bool is_strictly_increasing(ArrayRef<f64> grid) noexcept;

/// Returns true iff grid spacing is uniform within relative tolerance 1e-10.
[[nodiscard]] bool is_uniform_grid(ArrayRef<f64> grid) noexcept;

// ============================================================================
// 2. 1D Interpolation
// ============================================================================

enum class Interp1Method : u8 {
    Linear = 0,
    Nearest = 1,
    Previous = 2,
    Next = 3,
    Spline = 4,
    Pchip = 5,
    Makima = 6,
    Cubic = 7, // Keys cubic convolution on uniform grid, spline fallback otherwise
};

enum class ExtrapMode : u8 {
    Default = 0,  // Spline/Pchip/Makima extrapolate; other methods yield NaN out-of-range
    Method = 1,   // Extrapolate using the method
    Constant = 2, // Fill out-of-range queries with `extrap_val`
};

/// 1D interpolation on ascending knots `x` (length >= 2) and values `y`.
/// Writes `xq.size()` interpolated values into `out`. Returns false on invalid sizes.
[[nodiscard]] bool interp1(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> xq, MutArrayRef<f64> out,
                           Interp1Method method = Interp1Method::Linear,
                           ExtrapMode extrap = ExtrapMode::Default, f64 extrap_val = 0.0,
                           Allocator* scratch = nullptr);

// ============================================================================
// 3. Piecewise Polynomial Builders & Evaluation (PP-Form)
// ============================================================================

/// Builds not-a-knot cubic spline coefficients in column-major (pieces x 4) layout,
/// where pieces = x.size() - 1 and piece i evaluates a*u^3 + b*u^2 + c*u + d for u = xq - x[i].
[[nodiscard]] bool spline_pp_coefs(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> coefs_colmajor,
                                   Allocator* scratch = nullptr);

/// Builds shape-preserving PCHIP cubic coefficients in column-major (pieces x 4) layout.
[[nodiscard]] bool pchip_pp_coefs(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> coefs_colmajor,
                                  Allocator* scratch = nullptr);

/// Builds Modified Akima (makima) cubic coefficients in column-major (pieces x 4) layout.
[[nodiscard]] bool makima_pp_coefs(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> coefs_colmajor,
                                   Allocator* scratch = nullptr);

/// Evaluates a piecewise polynomial with `breaks` (length pieces + 1) and column-major
/// `coefs_colmajor` (pieces x order, highest power first) at query points `xq`.
[[nodiscard]] bool ppval(ArrayRef<f64> breaks, ArrayRef<f64> coefs_colmajor, usize pieces,
                         usize order, ArrayRef<f64> xq, MutArrayRef<f64> out) noexcept;

// ============================================================================
// 4. 2D & 3D Gridded Interpolation
// ============================================================================

enum class Interp2Method : u8 {
    Linear = 0,
    Nearest = 1,
    Cubic = 2,  // Keys bicubic convolution (requires uniform grid and rows,cols >= 3)
    Spline = 3, // Separable tensor-product not-a-knot cubic spline (2D only)
};

/// 2D pointwise gridded interpolation on column-major `v_colmajor` (rows x cols),
/// with x_grid (length cols) and y_grid (length rows).
[[nodiscard]] bool interp2_pointwise(ArrayRef<f64> v_colmajor, usize rows, usize cols,
                                     ArrayRef<f64> x_grid, ArrayRef<f64> y_grid, ArrayRef<f64> xq,
                                     ArrayRef<f64> yq, MutArrayRef<f64> out,
                                     Interp2Method method = Interp2Method::Linear,
                                     Allocator* scratch = nullptr);

/// 2D grid-expanded interpolation: evaluates at every (xq_cols[j], yq_rows[i]) into
/// column-major `out_colmajor` of shape (yq_rows.size() x xq_cols.size()).
[[nodiscard]] bool interp2_grid(ArrayRef<f64> v_colmajor, usize rows, usize cols,
                                ArrayRef<f64> x_grid, ArrayRef<f64> y_grid, ArrayRef<f64> xq_cols,
                                ArrayRef<f64> yq_rows, MutArrayRef<f64> out_colmajor,
                                Interp2Method method = Interp2Method::Linear,
                                Allocator* scratch = nullptr);

/// 3D pointwise gridded interpolation on column-major page-major `v_colmajor` (rows x cols x pages),
/// with x_grid (length cols), y_grid (length rows), z_grid (length pages).
/// Supports Linear and Nearest.
[[nodiscard]] bool interp3_pointwise(ArrayRef<f64> v_colmajor, usize rows, usize cols, usize pages,
                                     ArrayRef<f64> x_grid, ArrayRef<f64> y_grid,
                                     ArrayRef<f64> z_grid, ArrayRef<f64> xq, ArrayRef<f64> yq,
                                     ArrayRef<f64> zq, MutArrayRef<f64> out,
                                     Interp2Method method = Interp2Method::Linear) noexcept;

} // namespace keel::interp
