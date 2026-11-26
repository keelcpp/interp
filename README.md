# keel::interp

1D, 2D, and 3D gridded interpolation, cubic splines (not-a-knot, PCHIP, Modified Akima, Keys bicubic), and piecewise polynomial evaluation built on [`keel::keel`](https://github.com/keelcpp/keel).

## Features

- **1D Interpolation (`interp1`)**: `Linear`, `Nearest`, `Previous`, `Next`, `Spline` (not-a-knot), `Pchip` (shape-preserving Fritsch-Carlson), `Makima` (Modified Akima), `Cubic` (Keys $a = -0.5$ cubic convolution with spline fallback on non-uniform grids), with `Default` / `Method` / `Constant` extrapolation policies.
- **Piecewise Polynomial Forms**: `spline_pp_coefs`, `pchip_pp_coefs`, `makima_pp_coefs`, and `ppval`.
- **2D Gridded Interpolation**: `interp2_pointwise` and `interp2_grid` (`Linear`, `Nearest`, `Cubic`, separable `Spline`).
- **3D Gridded Interpolation**: `interp3_pointwise` (`Linear`, `Nearest`).
- **B-Spline & PP-Form Calculus**: `aveknt`, `augknt`, `brk2knt`, `knt2brk`, `ppval_nd`, `pp_deriv`, `pp_integ`.
- Zero STL in public headers, compiles cleanly under `-fno-exceptions -fno-rtti`.
- Differential reference test suite (`tests/verify_reference.py`) validated against SciPy and MATLAB.
