"""
stations.py
Part of the PropKot turbojet cycle analysis project.

Copyright (c) 2026 Andrii Kotenev
Licensed under the MIT License - see LICENSE file for details.
"""

"""
Here the calculations for each station of the engine are performed.

Each function takes the inlet conditions and component parameters,
and returns the exit conditions.

Notation follows MIT conventions:
  - Subscripts: 2=compressor inlet, 3=compressor exit, 4=turbine inlet, 5=turbine exit
  - T_t = stagnation temperature (K)
  - p_t = stagnation pressure (kPa)
  - h_t = stagnation enthalpy (kJ/kg)
  - pi  = pressure ratio
  - tau = temperature ratio
  - eta = isentropic efficiency
"""

import numpy as np
from properties import h, T_from_h

################################################

def compressor(T_t2, p_t2, pi_c, eta_c, gamma_c=1.4):
    """
    The compressor function, Args:
        T_t2: Compressor inlet stagnation temperature (K)
        p_t2: Compressor inlet stagnation pressure (kPa)
        pi_c: Compressor pressure ratio (p_t3/p_t2)
        eta_c: Compressor isentropic efficiency
        gamma_c: Cold-side specific heat ratio (default 1.4)
    """
    tau_c_ideal = pi_c ** ((gamma_c - 1) / gamma_c) 
    tau_c = 1 + (tau_c_ideal - 1) / eta_c
    T_t3 = tau_c * T_t2
    p_t3 = pi_c * p_t2
    h_t2 = h(T_t2, f=0)
    h_t3 = h(T_t3, f=0)
    work = h_t3 - h_t2 #work per kg in kJ/kg
    
    return{
        'T_t3': T_t3,
        'p_t3': p_t3,
        'h_t2': h_t2,
        'h_t3': h_t3,
        'tau_c': tau_c,
        'work': work,
    }

################################################


def combustor(p_t3, h_t3, T_t4, pi_b=0.95, eta_b=0.98, LHV=43000, max_iter=10, tol=1e-5):
    """
    Combustor funcction, Args:
        p_t3: Combustor inlet pressure (kPa)
        h_t3: Combustor inlet enthalpy (kJ/kg)
        T_t4: Turbine inlet temperature (K)
        pi_b: Combustor pressure ratio (default 0.95)
        eta_b: Combustor efficiency (default 0.98)
        LHV: Lower heating value of fuel kerosine based Jet-A
    """

    from properties import h_at_f

    p_t4 = pi_b * p_t3
    f = 0.020 #initial guess
    
    for i in range(max_iter):
        h_t4 = h_at_f(T_t4, f) #2d interpolation, ht4 at current T and f
        f_new = (h_t4 - h_t3) / (eta_b * LHV - h_t4)
        if abs(f_new - f) < tol:
            f = f_new
            break
        
        f = f_new
    
    # Recompute h_t4 at final f
    h_t4 = h_at_f(T_t4, f)
    
    return {
        'p_t4': p_t4,
        'h_t4': h_t4,
        'f': f,
        'iterations': i + 1,
    }

################################################

def turbine(T_t4, p_t4, h_t4, work_compressor, f, eta_t=0.74, eta_m=1.0):
    """
    Shaft balance to find h_t5 and T_t5, then pressure ratio
    using local gamma at the mean turbine temperature.
    
    Args:
        T_t4: Turbine inlet temperature (K)
        p_t4: Turbine inlet pressure (kPa)
        h_t4: Turbine inlet enthalpy (kJ/kg)
        work_compressor: Compressor specific work (kJ/kg)
        f: Fuel-air ratio
        eta_t: Turbine isentropic efficiency (default 0.74)
        eta_m: Mechanical efficiency (default 1.0)
    """
    from properties import T_from_h_at_f, gamma
    
    work_turbine = work_compressor / ((1 + f) * eta_m)
    h_t5 = h_t4 - work_turbine
    
    T_t5 = T_from_h_at_f(h_t5, f)
    
    tau_t = T_t5 / T_t4
    
    T_mean = (T_t4 + T_t5) / 2.0
    gamma_t = gamma(T_mean, f)
    
    tau_t_ideal = 1 - (1 - tau_t) / eta_t
    pi_t = tau_t_ideal ** (gamma_t / (gamma_t - 1))
    
    p_t5 = pi_t * p_t4
    
    return {
        'T_t5': T_t5,
        'p_t5': p_t5,
        'h_t5': h_t5,
        'tau_t': tau_t,
        'pi_t': pi_t,
        'gamma_t_used': gamma_t,
    }

################################################

def nozzle(T_t5, p_t5, h_t5, f, p_0=101.3, eta_n=1.0):
    """
    Handles both choked and unchoked cases (just in case).
    Args:
        T_t5: Turbine exit temperature (K)
        p_t5: Turbine exit pressure (kPa)
        h_t5: Turbine exit enthalpy (kJ/kg)
        f: Fuel-air ratio (for property lookup)
        p_0: Ambient pressure (kPa, default 101.3 = sea level)
        eta_n: Nozzle isentropic efficiency (default 1.0 for first-pass)
    """
    from properties import gamma, cp
    gamma_n = gamma(T_t5, f)
    cp_n = cp(T_t5, f) 
    
    NPR = p_t5 / p_0
    
    NPR_crit = ((gamma_n + 1) / 2) ** (gamma_n / (gamma_n - 1))
    
    choked = NPR > NPR_crit
    
    if choked:   
        p_exit = p_t5 / NPR_crit
        T_exit = T_t5 * 2 / (gamma_n + 1)
        R_local = cp_n * (gamma_n - 1) / gamma_n
        V_exit_ideal = np.sqrt(gamma_n * R_local * T_exit)
        M_exit = 1.0
    else:
        p_exit = p_0
        T_exit = T_t5 * (p_0 / p_t5) ** ((gamma_n - 1) / gamma_n)
        # Energy conservation: V² = 2·cp·(T_t5 - T_exit)
        V_exit_ideal = np.sqrt(2 * cp_n * (T_t5 - T_exit))
        # Exit Mach
        R_local = cp_n * (gamma_n - 1) / gamma_n
        a_exit = np.sqrt(gamma_n * R_local * T_exit)
        M_exit = V_exit_ideal / a_exit
    
    #nozzle efficiency
    V_exit = np.sqrt(eta_n) * V_exit_ideal
    
    return {
        'V_exit': V_exit,
        'M_exit': M_exit,
        'T_exit': T_exit,
        'p_exit': p_exit,
        'choked': choked,
        'NPR': NPR,
        'NPR_crit': NPR_crit,
        'gamma_n': gamma_n,
    }

################################################

'''

print("Compressor test (candidate 5: PR=3.4, η_c=0.72)")
comp = compressor(T_t2=288, p_t2=101.3, pi_c=3.4, eta_c=0.72)
    
print(f"T_t3  = {comp['T_t3']:.2f} K  ")
print(f"p_t3  = {comp['p_t3']:.2f} kPa ")
print(f"h_t2  = {comp['h_t2']:.2f} kJ/kg ")
print(f"h_t3  = {comp['h_t3']:.2f} kJ/kg ")
print(f"tau_c = {comp['tau_c']:.4f}  ")
print(f"work  = {comp['work']:.2f} kJ/kg")
print()

print("Combustor")
comb = combustor(p_t3=comp['p_t3'], h_t3=comp['h_t3'], T_t4=1250)
print(f"p_t4  = {comb['p_t4']:.2f} kPa")
print(f"h_t4  = {comb['h_t4']:.2f} kJ/kg ")
print(f"f     = {comb['f']:.5f}  ")
print(f"iter  = {comb['iterations']} ")
print()

print("Turbine")
turb = turbine(
    T_t4=1250, p_t4=comb['p_t4'], h_t4=comb['h_t4'],
    work_compressor=comp['work'], f=comb['f']
)
print(f"T_t5  = {turb['T_t5']:.2f} K")
print(f"p_t5  = {turb['p_t5']:.2f} kPa")
print(f"h_t5  = {turb['h_t5']:.2f} kJ/kg")
print(f"tau_t = {turb['tau_t']:.4f}")
print(f"pi_t  = {turb['pi_t']:.4f}")
print(f"gamma_t_used  = {turb['gamma_t_used']:.4f}")

print("\nNozzle")
noz = nozzle(
    T_t5=turb['T_t5'],
    p_t5=turb['p_t5'],
    h_t5=turb['h_t5'],
    f=comb['f']
)
print(f"NPR      = {noz['NPR']:.3f}  ")
print(f"NPR_crit = {noz['NPR_crit']:.3f}  ")
print(f"choked   = {noz['choked']}")
print(f"T_exit   = {noz['T_exit']:.2f} K")
print(f"M_exit   = {noz['M_exit']:.3f}")
print(f"V_exit   = {noz['V_exit']:.2f} m/s")
print(f"gamma_n  = {noz['gamma_n']:.4f}")

mdot_air = 0.907  
thrust = mdot_air * (1 + comb['f']) * noz['V_exit']
print(f"\nFinal thrust")
print(f"Mass flow (air) = {mdot_air:.3f} kg/s")
print(f"Thrust F = {thrust:.2f} N")
'''
