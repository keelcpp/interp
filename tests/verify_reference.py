#!/usr/bin/env python3
"""Differential reference verification for keel::interp against SciPy and MATLAB."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate


def parse_dump(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        vals = [float(tok) for tok in rest.strip().split()] if rest.strip() else []
        out[key.strip()] = vals
    return out


def check_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, failures: list[str]) -> None:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        failures.append(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    err = float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0
    status = "PASS" if err <= tol else "FAIL"
    print(f"  [{status}] {name:<28s} rel_err={err:.3e} (tol={tol:.1e})")
    if err > tol:
        failures.append(f"{name}: rel_err={err:.3e} > tol={tol:.1e}")


def run_matlab_checks(dump: dict[str, list[float]], failures: list[str]) -> None:
    if not shutil.which("matlab"):
        print("  [SKIP] MATLAB not found on PATH")
        return

    script = r"""
format longg
x_nu = [0 0.5 2.0 3.0 5.0];
y_nu = [0 1.2 -0.5 2.3 1.0];
xq_nu = [-0.5 0.25 1.1 2.7 4.2 5.5];
fprintf('m_spline_nu: %s\n', sprintf('%.17g ', interp1(x_nu, y_nu, xq_nu, 'spline')));
fprintf('m_pchip_nu: %s\n', sprintf('%.17g ', interp1(x_nu, y_nu, xq_nu, 'pchip')));
fprintf('m_makima_nu: %s\n', sprintf('%.17g ', interp1(x_nu, y_nu, xq_nu, 'makima')));

x_f = [0 1 2 3 4];
y_f = [1 1 5 6 6];
xq_f = [0.25 0.5 0.75 3.25 3.5 3.75];
fprintf('m_pchip_flat: %s\n', sprintf('%.17g ', interp1(x_f, y_f, xq_f, 'pchip')));
fprintf('m_makima_flat: %s\n', sprintf('%.17g ', interp1(x_f, y_f, xq_f, 'makima')));

xg = [0 0.7 1.8 3.0];
yg = [-1.0 0.2 1.5 2.5];
[XX, YY] = meshgrid(xg, yg);
V2 = sin(XX).*cos(YY) + 0.3*XX.*YY;
q2x = [0.35 1.2 2.4];
q2y = [-0.4 0.8 2.1];
fprintf('m_interp2_linear: %s\n', sprintf('%.17g ', interp2(xg, yg, V2, q2x, q2y, 'linear')));
fprintf('m_interp2_spline: %s\n', sprintf('%.17g ', interp2(xg, yg, V2, q2x, q2y, 'spline')));

ug = [0 1 2 3];
[UX, UY] = meshgrid(ug, ug);
V2u = sin(UX) + cos(UY);
qux = [0.4 1.3 2.6];
quy = [0.7 1.8 2.2];
fprintf('m_interp2_cubic: %s\n', sprintf('%.17g ', interp2(ug, ug, V2u, qux, quy, 'cubic')));

g3x = [0 1 3]; g3y = [-1 0.5 2]; g3z = [0 2];
[X3, Y3, Z3] = meshgrid(g3x, g3y, g3z);
V3 = 2*X3 - Y3 + 0.5*Z3 + X3.*Y3.*Z3;
fprintf('m_interp3_linear: %s\n', sprintf('%.17g ', interp3(g3x, g3y, g3z, V3, [0.5 2.2], [0.0 1.25], [0.8 1.6], 'linear')));
"""
    proc = subprocess.run(["matlab", "-batch", script], capture_output=True, text=True, check=True)
    md = parse_dump(proc.stdout)
    check_close("matlab/spline_nu", dump["spline_nu"], md["m_spline_nu"], 1e-13, failures)
    check_close("matlab/pchip_nu", dump["pchip_nu"], md["m_pchip_nu"], 1e-14, failures)
    check_close("matlab/makima_nu", dump["makima_nu"], md["m_makima_nu"], 1e-14, failures)
    check_close("matlab/pchip_flat", dump["pchip_flat"], md["m_pchip_flat"], 1e-15, failures)
    check_close("matlab/makima_flat", dump["makima_flat"], md["m_makima_flat"], 1e-14, failures)
    check_close("matlab/interp2_linear", dump["interp2_linear"], md["m_interp2_linear"], 1e-14, failures)
    check_close("matlab/interp2_spline", dump["interp2_spline"], md["m_interp2_spline"], 1e-13, failures)
    check_close("matlab/interp2_cubic", dump["interp2_cubic"], md["m_interp2_cubic"], 1e-13, failures)
    check_close("matlab/interp3_linear", dump["interp3_linear"], md["m_interp3_linear"], 1e-14, failures)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path-to-test_interp>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1])
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    dump = parse_dump(proc.stdout)
    failures: list[str] = []

    print("Running keel::interp reference verification against SciPy/MATLAB...")

    x_nu = np.array([0.0, 0.5, 2.0, 3.0, 5.0])
    y_nu = np.array([0.0, 1.2, -0.5, 2.3, 1.0])
    xq_nu = np.array([-0.5, 0.25, 1.1, 2.7, 4.2, 5.5])
    xq_in = np.array([0.25, 1.1, 2.7, 4.2])

    cs = interpolate.CubicSpline(x_nu, y_nu, bc_type="not-a-knot")
    pc = interpolate.PchipInterpolator(x_nu, y_nu)
    ak = interpolate.Akima1DInterpolator(x_nu, y_nu, method="makima")

    check_close("scipy/spline_nu", dump["spline_nu"], cs(xq_nu), 1e-13, failures)
    check_close("scipy/pchip_nu", dump["pchip_nu"], pc(xq_nu), 1e-14, failures)
    check_close("scipy/makima_nu", dump["makima_nu"], ak(xq_nu, extrapolate=True), 1e-14, failures)
    check_close("scipy/linear_nu", dump["linear_nu"], np.interp(xq_in, x_nu, y_nu), 1e-14, failures)

    x_f = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_f = np.array([1.0, 1.0, 5.0, 6.0, 6.0])
    xq_f = np.array([0.25, 0.5, 0.75, 3.25, 3.5, 3.75])
    check_close("scipy/pchip_flat", dump["pchip_flat"], interpolate.PchipInterpolator(x_f, y_f)(xq_f), 1e-15, failures)
    check_close(
        "scipy/makima_flat",
        dump["makima_flat"],
        interpolate.Akima1DInterpolator(x_f, y_f, method="makima")(xq_f),
        1e-14,
        failures,
    )

    # 2D RegularGridInterpolator linear
    xg = np.array([0.0, 0.7, 1.8, 3.0])
    yg = np.array([-1.0, 0.2, 1.5, 2.5])
    XX, YY = np.meshgrid(xg, yg, indexing="xy")
    V2 = np.sin(XX) * np.cos(YY) + 0.3 * XX * YY
    rgi2 = interpolate.RegularGridInterpolator((yg, xg), V2, method="linear")
    pts2 = np.column_stack([[-0.4, 0.8, 2.1], [0.35, 1.2, 2.4]])
    check_close("scipy/interp2_linear", dump["interp2_linear"], rgi2(pts2), 1e-14, failures)

    # 3D RegularGridInterpolator linear
    g3x = np.array([0.0, 1.0, 3.0])
    g3y = np.array([-1.0, 0.5, 2.0])
    g3z = np.array([0.0, 2.0])
    YY3, XX3, ZZ3 = np.meshgrid(g3y, g3x, g3z, indexing="ij")
    V3 = 2.0 * XX3 - YY3 + 0.5 * ZZ3 + XX3 * YY3 * ZZ3
    rgi3 = interpolate.RegularGridInterpolator((g3y, g3x, g3z), V3, method="linear")
    pts3 = np.column_stack([[0.0, 1.25], [0.5, 2.2], [0.8, 1.6]])
    check_close("scipy/interp3_linear", dump["interp3_linear"], rgi3(pts3), 1e-14, failures)

    # PP derivative & integral vs CubicSpline.derivative / antiderivative
    check_close("scipy/pp_deriv1", dump["pp_deriv1"], cs.derivative(1)(xq_in), 1e-13, failures)
    check_close("scipy/pp_integ", dump["pp_integ"], cs.antiderivative(1)(xq_in), 1e-13, failures)

    run_matlab_checks(dump, failures)

    if failures:
        print(f"\nFAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL keel::interp REFERENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
