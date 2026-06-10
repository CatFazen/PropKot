"""
Here I call stations.py to do a full cycle analysis. Provides sweep functions for parameter studies.
"""

import numpy as np
import matplotlib.pyplot as plt
from stations import compressor, combustor, turbine, nozzle

map_points = {
    'pi_c':       [2.6,     2.8,    3.0,    3.2,    3.4,    3.6],
    'eta_c':      [0.7375,   0.735,   0.73,  0.725,   0.72,   0.71],
    'mdot_lbmin': [100,     105,    110,    115,    120,    125],
}

map_points['mdot_kgs'] = [x / 132.28 for x in map_points['mdot_lbmin']]

########## interpolation of the compressor map

def compressor_map(pi_c):
    """
    Interpolate compressor map: given a pressure ratio, return eta_c and mdot.
    """
    from scipy.interpolate import interp1d
    
    if pi_c < min(map_points['pi_c']) or pi_c > max(map_points['pi_c']):
        raise ValueError(f"pi_c={pi_c} outside map range "
                         f"[{min(map_points['pi_c'])}, {max(map_points['pi_c'])}]")
    
    eta_interp = interp1d(map_points['pi_c'], map_points['eta_c'], kind='linear')
    mdot_interp = interp1d(map_points['pi_c'], map_points['mdot_kgs'], kind='linear')
    
    return float(eta_interp(pi_c)), float(mdot_interp(pi_c))

#########Sweep based on pi_c 

def sweep_pi_c(pi_c_values, T_t4=1250, **kwargs):
    """
    Sweep across compressor pressure ratio, runs the full cycle at each.
    """
    results = {
        'pi_c': [],
        'eta_c': [],
        'mdot_air': [],
        'thrust': [],
        'TSFC_kgf_hr': [],
        'T_t5': [],
        'V_exit': [],
        'M_exit': [],
        'f': [],
        'choked': [],
    }
    
    for pi_c in pi_c_values:
        try:
            eta_c, mdot = compressor_map(pi_c)
            res = run_cycle(pi_c=pi_c, eta_c=eta_c, mdot_air=mdot,
                            T_t4=T_t4, **kwargs)
            results['pi_c'].append(pi_c)
            results['eta_c'].append(eta_c)
            results['mdot_air'].append(mdot)
            results['thrust'].append(res['thrust'])
            results['TSFC_kgf_hr'].append(res['TSFC_kgf_hr'])
            results['T_t5'].append(res['T_t5'])
            results['V_exit'].append(res['V_exit'])
            results['M_exit'].append(res['M_exit'])
            results['f'].append(res['f'])
            results['choked'].append(res['choked'])
        except ValueError as e:
            print(f"Skipping pi_c={pi_c}: {e}")
    
    # Convert lists to numpy arrays for easier plotting
    for key in results:
        results[key] = np.array(results[key])
    
    return results

################# T_t4 sweep

def sweep_T_t4(T_t4_values, pi_c=3.4, **kwargs):
    """
    Sweep across turbine inlet temperature at fixed compressor operating point.
    
    Args:
        T_t4_values: array of T_t4 values to sweep (K)
        pi_c: compressor pressure ratio (default 3.4, design point)
        **kwargs: other parameters passed to run_cycle
    
    Returns:
        Dictionary with arrays of all output quantities.
    """
    eta_c, mdot = compressor_map(pi_c)
    
    results = {
        'T_t4': [],
        'thrust': [],
        'TSFC_kgf_hr': [],
        'T_t5': [],
        'V_exit': [],
        'f': [],
    }
    
    for T_t4 in T_t4_values:
        res = run_cycle(pi_c=pi_c, eta_c=eta_c, mdot_air=mdot,
                        T_t4=T_t4, **kwargs)
        results['T_t4'].append(T_t4)
        results['thrust'].append(res['thrust'])
        results['TSFC_kgf_hr'].append(res['TSFC_kgf_hr'])
        results['T_t5'].append(res['T_t5'])
        results['V_exit'].append(res['V_exit'])
        results['f'].append(res['f'])
    
    for key in results:
        results[key] = np.array(results[key])
    return results

############# Turbine efficiency sweep

def sweep_eta_t(eta_t_values, pi_c=3.4, T_t4=1250, **kwargs):
    """
    Sweep across turbine efficiency for thrust uncertainty band.
    """
    eta_c, mdot = compressor_map(pi_c)
    
    results = {
        'eta_t': [],
        'thrust': [],
        'TSFC_kgf_hr': [],
        'T_t5': [],
        'V_exit': [],
        'pi_t': [],
    }
    
    for eta_t in eta_t_values:
        res = run_cycle(pi_c=pi_c, eta_c=eta_c, mdot_air=mdot,
                        T_t4=T_t4, eta_t=eta_t, **kwargs)
        results['eta_t'].append(eta_t)
        results['thrust'].append(res['thrust'])
        results['TSFC_kgf_hr'].append(res['TSFC_kgf_hr'])
        results['T_t5'].append(res['T_t5'])
        results['V_exit'].append(res['V_exit'])
        results['pi_t'].append(res['pi_t'])
    
    for key in results:
        results[key] = np.array(results[key])
    return results

##############plot 

def plot_sweep(results, title="Cycle parameter sweep"):
    """
    Plot key quantities vs compressor pressure ratio from a sweep.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Thrust vs pi_c
    axes[0, 0].plot(results['pi_c'], results['thrust'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Compressor PR (π_c)')
    axes[0, 0].set_ylabel('Thrust (N)')
    axes[0, 0].set_title('Thrust vs Pressure Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(500, color='red', linestyle='--', alpha=0.5, label='500 N target')
    axes[0, 0].legend()
    
    # TSFC vs pi_c
    axes[0, 1].plot(results['pi_c'], results['TSFC_kgf_hr'], 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Compressor PR (π_c)')
    axes[0, 1].set_ylabel('TSFC (kg/(kgf·hr))')
    axes[0, 1].set_title('Fuel Consumption vs Pressure Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # V_exit vs pi_c
    axes[1, 0].plot(results['pi_c'], results['V_exit'], 'r-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Compressor PR (π_c)')
    axes[1, 0].set_ylabel('Nozzle Exit Velocity (m/s)')
    axes[1, 0].set_title('Exit Velocity vs Pressure Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # eta_c and mdot vs pi_c (dual axis)
    ax4 = axes[1, 1]
    ax4_b = ax4.twinx()
    ax4.plot(results['pi_c'], results['eta_c'], 'b-o', label='η_c')
    ax4_b.plot(results['pi_c'], results['mdot_air'], 'r-s', label='ṁ (kg/s)')
    ax4.set_xlabel('Compressor PR (π_c)')
    ax4.set_ylabel('η_c', color='blue')
    ax4_b.set_ylabel('Mass flow (kg/s)', color='red')
    ax4.set_title('Map: η_c and ṁ vs Pressure Ratio')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    #plt.show()

def plot_T_t4_sweep(results, title="Turbine inlet temperature sweep"):
    """Plot thrust and TSFC vs T_t4."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    axes[0].plot(results['T_t4'], results['thrust'], 'b-o', linewidth=2, markersize=6)
    axes[0].axvline(1250, color='red', linestyle='--', alpha=0.5, label='Design point')
    axes[0].set_xlabel('Turbine inlet temperature T_t4 (K)')
    axes[0].set_ylabel('Thrust (N)')
    axes[0].set_title('Thrust vs T_t4')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(results['T_t4'], results['TSFC_kgf_hr'], 'g-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Turbine inlet temperature T_t4 (K)')
    axes[1].set_ylabel('TSFC (kg/(kgf·hr))')
    axes[1].set_title('TSFC vs T_t4')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    #plt.show()


def plot_eta_t_sweep(results, title="Turbine efficiency sensitivity"):
    """Plot thrust uncertainty band from η_t variation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    idx_baseline = np.argmin(np.abs(results['eta_t'] - 0.74))
    baseline_thrust = results['thrust'][idx_baseline]
    
    axes[0].plot(results['eta_t'], results['thrust'], 'b-o', linewidth=2, markersize=6)
    axes[0].axvline(0.74, color='red', linestyle='--', alpha=0.5, 
                    label=f'Baseline (0.74): {baseline_thrust:.0f} N')
    axes[0].axvspan(0.70, 0.78, alpha=0.15, color='gray', label='Literature range 0.70–0.78')
    axes[0].set_xlabel('Turbine efficiency η_t')
    axes[0].set_ylabel('Thrust (N)')
    axes[0].set_title('Thrust sensitivity to η_t')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(results['eta_t'], results['pi_t'], 'r-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Turbine efficiency η_t')
    axes[1].set_ylabel('Turbine pressure ratio π_t')
    axes[1].set_title('π_t vs η_t')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    #plt.show()

######### run the whole thing

def run_cycle(pi_c, eta_c, mdot_air, T_t4=1250, T_t2=288, p_t2=101.3,        #eta_n (default 0.97, typical for convergent nozzles)    eta_m (default 0.99, bearing losses)
              pi_b=0.95, eta_b=0.98, eta_t=0.74, eta_n=0.97, eta_m=0.99,
              LHV=43000): 
    """
    Run the full turbojet cycle for one design point.
    """
    #stations
    comp = compressor(T_t2=T_t2, p_t2=p_t2, pi_c=pi_c, eta_c=eta_c)
    comb = combustor(
        p_t3=comp['p_t3'], h_t3=comp['h_t3'], T_t4=T_t4,
        pi_b=pi_b, eta_b=eta_b, LHV=LHV
    )

    turb = turbine(
        T_t4=T_t4, p_t4=comb['p_t4'], h_t4=comb['h_t4'],
        work_compressor=comp['work'], f=comb['f'],
        eta_t=eta_t, eta_m=eta_m
    )

    noz = nozzle(
        T_t5=turb['T_t5'], p_t5=turb['p_t5'], h_t5=turb['h_t5'],
        f=comb['f'], eta_n=eta_n
    )

    #Perf 
    mdot_fuel = mdot_air * comb['f']                    
    mdot_total = mdot_air + mdot_fuel                   # kg/s through turbine + nozzle
    thrust = mdot_total * noz['V_exit']                 # N
    TSFC_si = mdot_fuel / thrust                        # kg/(N·s), only for static
    TSFC_kgf_hr = TSFC_si * 9.81 * 3600                 # kg/(kgf·hr) 
    specific_thrust = thrust / mdot_air 
    Isp = 3600 / TSFC_kgf_hr
    from properties import gamma
    gamma_cold = gamma(T_t2, f=0)
    eta_th_ideal = 1 - 1 / pi_c**((gamma_cold - 1) / gamma_cold)
    q_in = LHV * comb['f'] / (1 + comb['f'])  #heat in mixture
    q_in_J = q_in * 1000
    ke_out = noz['V_exit']**2 / 2  #kinetic en
    eta_th_actual = ke_out / q_in_J #thermal efficiency 


    result = {
        'pi_c': pi_c, 'eta_c': eta_c, 'mdot_air': mdot_air, 'T_t4': T_t4,
        'T_t2': T_t2, 'p_t2': p_t2, 'h_t2': comp['h_t2'],
        'T_t3': comp['T_t3'], 'p_t3': comp['p_t3'], 'h_t3': comp['h_t3'],
        'T_t4': T_t4, 'p_t4': comb['p_t4'], 'h_t4': comb['h_t4'],
        'T_t5': turb['T_t5'], 'p_t5': turb['p_t5'], 'h_t5': turb['h_t5'],
        'tau_c': comp['tau_c'], 'tau_t': turb['tau_t'],
        'pi_t': turb['pi_t'],
        'f': comb['f'],
        'V_exit': noz['V_exit'], 'M_exit': noz['M_exit'],
        'choked': noz['choked'],
        'thrust': thrust,
        'TSFC_si': TSFC_si,
        'TSFC_kgf_hr': TSFC_kgf_hr,
        'specific_thrust': specific_thrust,
        'mdot_fuel': mdot_fuel,
        'Isp': Isp,
        'eta_th_ideal': eta_th_ideal,
        'eta_th_actual': eta_th_actual,
    }
    return result


################# Result table

def print_summary(result):
    print("=" * 60)
    print(f"Design point: π_c={result['pi_c']}, η_c={result['eta_c']}, "
      f"ṁ={result['mdot_air']:.4f} kg/s, T_t4={result['T_t4']} K")
    print("=" * 60)
    print(f"{'Station':<10} {'T_t (K)':>10} {'p_t (kPa)':>12} {'h_t (kJ/kg)':>14}")
    print("-" * 60)
    print(f"{'2 (inlet)':<10} {result['T_t2']:>10.1f} {result['p_t2']:>12.1f} {result['h_t2']:>14.2f}")
    print(f"{'3 (cmp ex)':<10} {result['T_t3']:>10.1f} {result['p_t3']:>12.1f} {result['h_t3']:>14.2f}")
    print(f"{'4 (trb in)':<10} {result['T_t4']:>10.1f} {result['p_t4']:>12.1f} {result['h_t4']:>14.2f}")
    print(f"{'5 (trb ex)':<10} {result['T_t5']:>10.1f} {result['p_t5']:>12.1f} {result['h_t5']:>14.2f}")
    print()
    print(f"Compressor temp ratio τ_c  = {result['tau_c']:.4f}")
    print(f"Turbine temp ratio τ_t     = {result['tau_t']:.4f}")
    print(f"Turbine pressure ratio π_t = {result['pi_t']:.4f}")
    print(f"Fuel-air ratio f           = {result['f']:.5f}")
    print()
    print(f"Nozzle exit velocity V     = {result['V_exit']:.1f} m/s")
    print(f"Nozzle exit Mach M         = {result['M_exit']:.3f}")
    print(f"Choked nozzle?             = {result['choked']}")
    print()
    print(f"Thrust F                   = {result['thrust']:.1f} N")
    print(f"Specific thrust F/ṁ        = {result['specific_thrust']:.1f} N·s/kg")
    print(f"Fuel flow ṁ_f              = {result['mdot_fuel']:.4f} kg/s "
          f"({result['mdot_fuel']*3600:.1f} kg/hr)")
    print(f"TSFC                       = {result['TSFC_kgf_hr']:.3f} kg/(kgf·hr)")
    print(f"Specific impulse Isp       = {result['Isp']:.0f} s")
    print(f"Brayton η_th ideal        = {result['eta_th_ideal']*100:.1f}%")
    print(f"Brayton η_th actual       = {result['eta_th_actual']*100:.1f}%")
    print("=" * 60)

############## Output

if __name__ == "__main__":

    print("\nDESIGN POINT")
    eta_c, mdot = compressor_map(3.4)
    result = run_cycle(pi_c=3.4, eta_c=eta_c, mdot_air=mdot, T_t4=1250)
    print_summary(result)
    
    
    print("\n PRESSURE RATIO SWEEP")
    pi_c_array = np.linspace(2.6, 3.6, 41)
    sweep_pr = sweep_pi_c(pi_c_array, T_t4=1250)

    print(f"\n{'π_c':<6} {'η_c':<7} {'ṁ (kg/s)':<10} {'T_t5 (K)':<10} {'V_exit':<10} {'Thrust (N)':<12} {'TSFC':<8} {'f':<8}")
    for i in range(len(sweep_pr['pi_c'])):
        print(f"{sweep_pr['pi_c'][i]:<6.2f} {sweep_pr['eta_c'][i]:<7.3f} "
            f"{sweep_pr['mdot_air'][i]:<10.3f} {sweep_pr['T_t5'][i]:<10.1f} "
            f"{sweep_pr['V_exit'][i]:<10.2f} {sweep_pr['thrust'][i]:<12.1f} "
            f"{sweep_pr['TSFC_kgf_hr'][i]:<8.3f} {sweep_pr['f'][i]:<8.5f}")

    plot_sweep(sweep_pr, title="S400SX-E Turbojet — π_c sweep, T_t4=1250 K")
    

    print("\nT_t4 SWEEP")
    T_t4_array = np.linspace(1100, 1300, 41)
    sweep_t4 = sweep_T_t4(T_t4_array, pi_c=3.4)
    print(f"\n{'T_t4 (K)':<10} {'T_t5 (K)':<10} {'V_exit':<10} {'Thrust (N)':<12} {'TSFC':<8} {'f':<8}")
    for i in range(len(sweep_t4['T_t4'])):
        print(f"{sweep_t4['T_t4'][i]:<10.1f} {sweep_t4['T_t5'][i]:<10.1f} "
            f"{sweep_t4['V_exit'][i]:<10.2f} {sweep_t4['thrust'][i]:<12.1f} "
            f"{sweep_t4['TSFC_kgf_hr'][i]:<8.3f} {sweep_t4['f'][i]:<8.5f}")
    plot_T_t4_sweep(sweep_t4, title="Thrust sensitivity to T_t4 (π_c = 3.4)")
    
    
    print("\n η_t UNCERTAINTY")
    eta_t_array = np.linspace(0.66, 0.82, 33)
    sweep_eta = sweep_eta_t(eta_t_array, pi_c=3.4, T_t4=1250)
    
    
    print(f"\n{'η_t':<7} {'π_t':<8} {'T_t5 (K)':<10} {'V_exit':<10} {'Thrust (N)':<12} {'TSFC':<8}")
    for i in range(len(sweep_eta['eta_t'])):
        print(f"{sweep_eta['eta_t'][i]:<7.3f} {sweep_eta['pi_t'][i]:<8.4f} "
             f"{sweep_eta['T_t5'][i]:<10.1f} {sweep_eta['V_exit'][i]:<10.2f} "
             f"{sweep_eta['thrust'][i]:<12.1f} {sweep_eta['TSFC_kgf_hr'][i]:<8.3f}")
    plot_eta_t_sweep(sweep_eta, title="η_t sensitivity (π_c = 3.4, T_t4 = 1250 K)")
    #plt.show()
