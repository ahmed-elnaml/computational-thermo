"""
================================================================================
 Computational Thermodynamics of a Binary Eutectic System
================================================================================
 CALPHAD assessment using pycalphad.
 - Automatically detects any TDB database files in the current folder.
 - Interactive free-energy curves with correct common-tangent construction.
 - Temperature indicator overlaid directly on the phase diagram.
 - Enthalpy, Entropy, and Activity plots with a temperature slider.

 HOW TO RUN (no Jupyter needed):
   python eutectic_thermo.py

 DEPENDENCIES:
   pip install pycalphad matplotlib numpy scipy
================================================================================
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # works on most desktops without Jupyter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons
from scipy.spatial import ConvexHull
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ── pycalphad ─────────────────────────────────────────────────────────────────
try:
    from pycalphad import Database, binplot, calculate, equilibrium, variables as v
except ImportError:
    print("\n[ERROR] pycalphad is not installed.")
    print("  Please run:  pip install pycalphad\n")
    sys.exit(1)

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 11,
    'axes.labelsize' : 13,
    'axes.titlesize' : 14,
    'lines.linewidth': 2.0,
    'figure.dpi'     : 110,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATABASE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

# Built-in synthetic Ag-Cu eutectic database (no external file needed)
_BUILTIN_NAME = "[Built-in] Ag-Cu synthetic eutectic"
_BUILTIN_TDB  = """\
$ Synthesized Ag-Cu Eutectic System
ELEMENT AG STANDARD  107.868 4260.32 42.55 !
ELEMENT CU STANDARD   63.546 3344.0  33.15 !

$ Reference states
FUNCTION UN_ASS  298.15 0.0; 3000.0 N !

$ Liquid Phase Definition
PHASE LIQUID:L %  1  1.0  !
CONSTITUENT LIQUID:L :AG,CU :  !
PARAMETER G(LIQUID,AG;0)  298.15  +11500-9.3*T; 3000 N !
PARAMETER G(LIQUID,CU;0)  298.15  +13000-9.6*T; 3000 N !
PARAMETER G(LIQUID,AG,CU;0)  298.15  +12000-3.0*T; 3000 N !

$ Solid Phase Definition (FCC)
PHASE FCC_A1 %  2  1.0  1.0  !
CONSTITUENT FCC_A1 :AG,CU : VA : !
PARAMETER G(FCC_A1,AG:VA;0) 298.15  0.0; 3000 N !
PARAMETER G(FCC_A1,CU:VA;0) 298.15  0.0; 3000 N !
PARAMETER G(FCC_A1,AG,CU:VA;0) 298.15 +28000-4.0*T; 3000 N !
"""

def _find_tdb_files():
    """Return a list of .tdb files in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(script_dir, '*.tdb')))


def _probe_database(dbf):
    """
    Inspect a Database object and return (comps, phases, second_element, T_range).
    Works for 1- or 2-component systems.
    """
    # Components (skip VA and /-)
    all_comps = [c for c in dbf.elements if c not in ('VA', '/-')]
    comps = sorted(all_comps) + ['VA']

    # Phases actually present in the DB
    phases = list(dbf.phases.keys())

    # Second element for composition axis (first non-trivial element)
    if len(all_comps) >= 2:
        second_el = sorted(all_comps)[1]   # e.g. 'CU' or 'ZN'
    else:
        second_el = all_comps[0]

    # Temperature range: safe defaults
    T_min, T_max = 500, 1600
    return comps, phases, second_el, T_min, T_max


def select_database_interactive():
    """
    Print a numbered menu of available databases (built-in + any .tdb files)
    and return (dbf, comps, phases, second_el, T_min, T_max, label).
    """
    tdb_files = _find_tdb_files()

    options = [_BUILTIN_NAME] + tdb_files
    print("\n" + "═"*60)
    print("  Computational Thermodynamics — Database Selection")
    print("═"*60)
    for idx, opt in enumerate(options):
        tag = os.path.basename(opt) if opt != _BUILTIN_NAME else opt
        print(f"  [{idx}]  {tag}")
    print("═"*60)

    while True:
        try:
            choice = input(f"  Select database (0–{len(options)-1}): ").strip()
            idx = int(choice)
            if 0 <= idx < len(options):
                break
            print(f"  Please enter a number between 0 and {len(options)-1}.")
        except ValueError:
            print("  Invalid input — please enter a number.")

    selected = options[idx]
    if selected == _BUILTIN_NAME:
        label = "Ag-Cu (built-in)"
        dbf   = Database(_BUILTIN_TDB)
    else:
        label = os.path.basename(selected)
        print(f"\n  Loading: {label}  ...")
        dbf = Database(selected)

    comps, phases, second_el, T_min, T_max = _probe_database(dbf)

    print(f"\n  ✔  Database loaded:  {label}")
    print(f"     Components : {comps}")
    print(f"     Phases     : {phases}")
    print(f"     Comp. axis : X({second_el})\n")

    return dbf, comps, phases, second_el, T_min, T_max, label


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMMON-TANGENT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_common_tangents(x_liq, g_liq, x_fcc, g_fcc):
    """
    Find all common-tangent pairs between the LIQUID and FCC_A1 curves.

    Returns a list of (x1, g1, x2, g2) tuples — the two contact points of
    each tangent line.  Uses a grid-search + Brent refinement approach that
    is robust even for noisy or multi-valued curves.
    """
    tangents = []

    # Remove NaN/Inf
    mask_l = np.isfinite(g_liq)
    mask_f = np.isfinite(g_fcc)
    xl, gl = x_liq[mask_l], g_liq[mask_l]
    xf, gf = x_fcc[mask_f], g_fcc[mask_f]

    if len(xl) < 4 or len(xf) < 4:
        return tangents

    from scipy.interpolate import interp1d
    interp_l = interp1d(xl, gl, kind='cubic', bounds_error=False, fill_value=np.nan)
    interp_f = interp1d(xf, gf, kind='cubic', bounds_error=False, fill_value=np.nan)

    # Search over pairs of compositions (one on each curve)
    # Tangent condition: slope of secant == derivative on both sides
    dx = 1e-4

    def slope_diff(xa, xb):
        """Secant slope minus local derivatives on both curves (combined residual)."""
        ga = interp_l(xa)
        gb = interp_f(xb)
        if np.isnan(ga) or np.isnan(gb):
            return np.nan
        secant = (gb - ga) / (xb - xa + 1e-12)
        dga = (interp_l(xa + dx) - interp_l(xa - dx)) / (2 * dx)
        dgb = (interp_f(xb + dx) - interp_f(xb - dx)) / (2 * dx)
        return 0.5 * ((dga - secant)**2 + (dgb - secant)**2)

    # Coarse grid scan
    n_grid = 80
    xa_vals = np.linspace(xl.min() + 0.01, xl.max() - 0.01, n_grid)
    xb_vals = np.linspace(xf.min() + 0.01, xf.max() - 0.01, n_grid)

    # For each xa try to minimise over xb
    from scipy.optimize import minimize_scalar

    candidates = []
    for xa in xa_vals:
        def obj(xb):
            if xb <= xa:
                return 1e10
            v = slope_diff(xa, xb)
            return v if not np.isnan(v) else 1e10

        res = minimize_scalar(obj, bounds=(xa + 0.01, xb_vals.max()), method='bounded')
        if res.fun < 1e4:
            candidates.append((xa, res.x, res.fun))

    if not candidates:
        return tangents

    # Sort by residual and keep the best few, then refine
    candidates.sort(key=lambda c: c[2])
    seen = []

    for xa0, xb0, _ in candidates[:10]:
        # Refine with a 2D minimizer
        from scipy.optimize import minimize
        res2 = minimize(lambda p: slope_diff(p[0], p[1]),
                        [xa0, xb0],
                        method='Nelder-Mead',
                        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
        xa_r, xb_r = res2.x
        if not (xl.min() < xa_r < xl.max() and xf.min() < xb_r < xf.max()):
            continue
        if xa_r >= xb_r - 0.005:
            continue

        # Deduplicate
        duplicate = any(abs(xa_r - s[0]) < 0.03 and abs(xb_r - s[1]) < 0.03
                        for s in seen)
        if not duplicate:
            ga_r = float(interp_l(xa_r))
            gb_r = float(interp_f(xb_r))
            if np.isfinite(ga_r) and np.isfinite(gb_r):
                seen.append((xa_r, xb_r))
                tangents.append((xa_r, ga_r, xb_r, gb_r))

    return tangents


def _find_fcc_fcc_tangent(x_fcc, g_fcc):
    """
    Find the common tangent between the two sides of the FCC miscibility gap
    (if it exists — i.e., the curve is not convex).
    """
    from scipy.interpolate import interp1d
    mask = np.isfinite(g_fcc)
    xf, gf = x_fcc[mask], g_fcc[mask]
    if len(xf) < 6:
        return None

    # Build convex hull on lower side to detect non-convexity
    pts = np.column_stack([xf, gf])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None

    # Lower hull vertices
    lower = sorted([hull.vertices[i] for i in range(len(hull.vertices))
                    if pts[hull.vertices[i], 1] <= np.median(pts[:, 1])],
                   key=lambda i: pts[i, 0])

    if len(lower) < 2:
        return None

    interp = interp1d(xf, gf, kind='cubic', bounds_error=False, fill_value=np.nan)
    dx = 1e-4

    from scipy.optimize import minimize

    def obj(p):
        xa, xb = p
        if xa <= 0.01 or xb >= 0.99 or xb - xa < 0.05:
            return 1e10
        ga = interp(xa)
        gb = interp(xb)
        if np.isnan(ga) or np.isnan(gb):
            return 1e10
        sec = (gb - ga) / (xb - xa)
        dga = (interp(xa + dx) - interp(xa - dx)) / (2*dx)
        dgb = (interp(xb + dx) - interp(xb - dx)) / (2*dx)
        return (dga - sec)**2 + (dgb - sec)**2 + (dga - dgb)**2

    best = None
    for xa0 in np.linspace(0.1, 0.4, 8):
        for xb0 in np.linspace(0.6, 0.9, 8):
            res = minimize(obj, [xa0, xb0], method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000})
            if res.fun < 1e-3:
                xa_r, xb_r = res.x
                if 0 < xa_r < xb_r < 1 and xb_r - xa_r > 0.05:
                    ga_r = float(interp(xa_r))
                    gb_r = float(interp(xb_r))
                    if np.isfinite(ga_r) and np.isfinite(gb_r):
                        if best is None or res.fun < best[4]:
                            best = (xa_r, ga_r, xb_r, gb_r, res.fun)
    if best:
        return best[:4]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. FREE-ENERGY PLOT (with common tangent)
# ══════════════════════════════════════════════════════════════════════════════

def _update_free_energy(T, dbf, comps, phases, second_el, ax_ge, ax_pd,
                        T_min, T_max, label, temp_line_ref):
    """Redraws the G-x panel and updates the temperature indicator on the phase diagram."""
    ax_ge.cla()

    # ── pycalphad calculations ─────────────────────────────────────────────
    phase_colors = {
        'LIQUID'  : '#2196F3',
        'FCC_A1'  : '#F44336',
        'HCP_A3'  : '#4CAF50',
        'BCC_A2'  : '#FF9800',
        'HCP_ZN'  : '#9C27B0',
    }
    default_colors = plt.cm.tab10.colors

    x_data = {}
    g_data = {}
    color_map = {}
    cidx = 0

    for ph in phases:
        try:
            res = calculate(dbf, comps, ph, P=101325, T=T, output='GM',
                            points_per_phase=200)
            xv = res.X.sel(component=second_el).values.flatten()
            gv = res.GM.values.flatten()

            # Sort by composition
            order = np.argsort(xv)
            xv, gv = xv[order], gv[order]

            mask = np.isfinite(gv)
            if mask.sum() < 3:
                continue

            x_data[ph] = xv
            g_data[ph] = gv
            col = phase_colors.get(ph, default_colors[cidx % len(default_colors)])
            color_map[ph] = col
            cidx += 1

            ax_ge.plot(xv[mask], gv[mask], color=col, label=ph, zorder=3)
        except Exception:
            pass

    # ── Common tangents ────────────────────────────────────────────────────
    tangent_drawn = False

    # Liquid ↔ Solid tangent(s)
    liq_phases = [p for p in x_data if 'LIQUID' in p.upper()]
    sol_phases  = [p for p in x_data if 'LIQUID' not in p.upper()]

    for lp in liq_phases:
        for sp in sol_phases:
            tans = _compute_common_tangents(x_data[lp], g_data[lp],
                                            x_data[sp], g_data[sp])
            for (x1, g1, x2, g2) in tans:
                ax_ge.plot([x1, x2], [g1, g2],
                           'k--', linewidth=1.6, zorder=5,
                           label='Common tangent' if not tangent_drawn else '')
                ax_ge.plot([x1], [g1], 'ko', markersize=7, zorder=6)
                ax_ge.plot([x2], [g2], 'ko', markersize=7, zorder=6)
                tangent_drawn = True

    # FCC ↔ FCC miscibility gap tangent
    for sp in sol_phases:
        result = _find_fcc_fcc_tangent(x_data[sp], g_data[sp])
        if result:
            x1, g1, x2, g2 = result
            ax_ge.plot([x1, x2], [g1, g2],
                       color=color_map[sp], linestyle='--', linewidth=1.6,
                       zorder=5,
                       label=f'{sp} miscibility gap' if not tangent_drawn else '')
            ax_ge.plot([x1, x2], [g1, g2], 'o',
                       color=color_map[sp], markersize=7, zorder=6)
            tangent_drawn = True

    # ── Formatting ─────────────────────────────────────────────────────────
    ax_ge.set_title(f'Molar Gibbs Free Energy  —  T = {T:.0f} K', fontweight='bold')
    ax_ge.set_xlabel(f'Mole Fraction of {second_el}  ($X_{{{second_el}}}$)')
    ax_ge.set_ylabel(r'$G_m$ (J/mol)')
    ax_ge.set_xlim(-0.02, 1.02)
    ax_ge.legend(loc='best', fontsize=9)
    ax_ge.grid(True, alpha=0.25)

    # ── Temperature indicator on phase diagram ─────────────────────────────
    if temp_line_ref[0] is not None:
        try:
            temp_line_ref[0].remove()
        except Exception:
            pass
    temp_line_ref[0] = ax_pd.axhline(y=T, color='gold', linewidth=2.2,
                                      linestyle='--', zorder=10,
                                      label=f'T = {T:.0f} K')
    # Refresh legend on phase diagram
    ax_pd.legend(loc='upper right', fontsize=8)

    plt.draw()


# ══════════════════════════════════════════════════════════════════════════════
# 4. THERMO PROPERTIES PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _update_thermo(T, dbf, comps, phases, second_el,
                   ax_h, ax_s, ax_a):
    for ax in [ax_h, ax_s, ax_a]:
        ax.cla()

    conds = {v.X(second_el): (0.01, 0.99, 0.02), v.T: T, v.P: 101325}

    try:
        eq_res = equilibrium(dbf, comps, phases, conds)
        x_el   = eq_res.X.sel(component=second_el).values.flatten()
        enthalpy = eq_res.HM.values.flatten()
        entropy  = eq_res.SM.values.flatten()
        mu_el    = eq_res.MU.sel(component=second_el).values.flatten()

        # Reference: pure second element in the first solid phase available
        sol = [p for p in phases if 'LIQUID' not in p.upper()]
        ref_phase = sol[0] if sol else phases[0]
        ref_conds = {v.X(second_el): 1.0, v.T: T, v.P: 101325}
        ref_res   = equilibrium(dbf, comps, [ref_phase], ref_conds)
        mu_ref    = float(ref_res.MU.sel(component=second_el).values.flatten()[0])

        R = 8.314
        activity = np.exp((mu_el - mu_ref) / (R * T))

        ax_h.plot(x_el, enthalpy, color='#9C27B0')
        ax_h.set_title('Global Enthalpy of Mixing', fontweight='bold')
        ax_h.set_xlabel(f'$X_{{{second_el}}}$')
        ax_h.set_ylabel(r'$H_m$ (J/mol)')
        ax_h.grid(True, alpha=0.25)

        ax_s.plot(x_el, entropy, color='#009688')
        ax_s.set_title('Global Entropy of Mixing', fontweight='bold')
        ax_s.set_xlabel(f'$X_{{{second_el}}}$')
        ax_s.set_ylabel(r'$S_m$ (J/K·mol)')
        ax_s.grid(True, alpha=0.25)

        ax_a.plot(x_el, activity, color='#FF5722', label='Calculated')
        ax_a.plot([0, 1], [0, 1], 'k--', label="Ideal (Raoult's)")
        ax_a.set_title(f'Thermodynamic Activity of {second_el}', fontweight='bold')
        ax_a.set_xlabel(f'$X_{{{second_el}}}$')
        ax_a.set_ylabel(f'$a_{{{second_el}}}$')
        ax_a.legend(fontsize=9)
        ax_a.set_ylim(-0.05, 1.05)
        ax_a.grid(True, alpha=0.25)

    except Exception as err:
        for ax in [ax_h, ax_s, ax_a]:
            ax.text(0.5, 0.5, f'Calculation error:\n{err}',
                    ha='center', va='center', transform=ax.transAxes,
                    color='red', fontsize=9)
    plt.draw()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Database selection ────────────────────────────────────────────────
    dbf, comps, phases, second_el, T_min, T_max, db_label = select_database_interactive()

    T_init = (T_min + T_max) / 2

    # ── Build the phase diagram (static, computed once) ───────────────────
    print("  Computing phase diagram … (this may take 10–30 s)")
    fig_pd, ax_pd = plt.subplots(figsize=(7, 5))
    try:
        binplot(dbf, comps, phases,
                {v.X(second_el): (0, 1, 0.02),
                 v.T: (T_min, T_max, 10),
                 v.P: 101325},
                ax=ax_pd)
    except Exception as err:
        ax_pd.text(0.5, 0.5, f'Phase diagram error:\n{err}',
                   ha='center', va='center', transform=ax_pd.transAxes,
                   color='red')
    ax_pd.set_title(f'Binary Phase Diagram  —  {db_label}', fontweight='bold')
    ax_pd.set_xlabel(f'Mole Fraction {second_el}')
    ax_pd.set_ylabel('Temperature (K)')
    ax_pd.set_ylim(T_min, T_max)
    fig_pd.tight_layout()

    # Mutable container so the callback can update the line reference
    temp_line_ref = [None]

    # ── Figure 2: Free-energy curves (interactive) ─────────────────────────
    fig_ge = plt.figure(figsize=(9, 6))
    gs_ge  = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.45,
                                figure=fig_ge)
    ax_ge      = fig_ge.add_subplot(gs_ge[0])
    ax_sl_ge   = fig_ge.add_subplot(gs_ge[1])
    fig_ge.subplots_adjust(bottom=0.12)

    sl_ge = Slider(ax_sl_ge, 'Temperature (K)', T_min, T_max,
                   valinit=T_init, valstep=25, color='steelblue')

    def on_temp_ge(val):
        _update_free_energy(sl_ge.val, dbf, comps, phases, second_el,
                            ax_ge, ax_pd, T_min, T_max, db_label, temp_line_ref)

    sl_ge.on_changed(on_temp_ge)

    # ── Figure 3: Thermo properties (interactive) ─────────────────────────
    fig_tp = plt.figure(figsize=(16, 5))
    gs_tp  = gridspec.GridSpec(2, 3, height_ratios=[5, 1], hspace=0.45,
                                figure=fig_tp)
    ax_h   = fig_tp.add_subplot(gs_tp[0, 0])
    ax_s   = fig_tp.add_subplot(gs_tp[0, 1])
    ax_a   = fig_tp.add_subplot(gs_tp[0, 2])
    ax_sl_tp = fig_tp.add_subplot(gs_tp[1, :])
    fig_tp.subplots_adjust(bottom=0.15)
    fig_tp.suptitle(f'Thermodynamic Properties  —  {db_label}', fontweight='bold')

    sl_tp = Slider(ax_sl_tp, 'Temperature (K)', T_min, T_max,
                   valinit=T_init, valstep=50, color='steelblue')

    def on_temp_tp(val):
        _update_thermo(sl_tp.val, dbf, comps, phases, second_el,
                       ax_h, ax_s, ax_a)

    sl_tp.on_changed(on_temp_tp)

    # ── Initial draws ──────────────────────────────────────────────────────
    print("  Drawing initial plots …")
    _update_free_energy(T_init, dbf, comps, phases, second_el,
                        ax_ge, ax_pd, T_min, T_max, db_label, temp_line_ref)
    fig_ge.canvas.draw_idle()
    fig_pd.canvas.draw_idle()

    _update_thermo(T_init, dbf, comps, phases, second_el, ax_h, ax_s, ax_a)
    fig_tp.canvas.draw_idle()

    print("\n  ✔  All windows are open.")
    print("  Use the sliders to explore the thermodynamics.\n")

    plt.show()


if __name__ == '__main__':
    main()