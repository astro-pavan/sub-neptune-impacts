import numpy as np
import matplotlib.pyplot as plt
import woma

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg

def plot_spherical_profiles(planet):    
    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    
    ax[0, 0].plot(planet.A1_r / R_earth, planet.A1_rho)
    ax[0, 0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0, 0].set_ylabel(r"Density, $\rho$ [kg m$^{-3}$]")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlim(0, None)
    
    ax[1, 0].plot(planet.A1_r / R_earth, planet.A1_m_enc / M_earth)
    ax[1, 0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[1, 0].set_ylabel(r"Enclosed Mass, $M_{<r}$ $[M_\oplus]$")
    ax[1, 0].set_xlim(0, None)
    ax[1, 0].set_ylim(0, None)
    
    ax[0, 1].plot(planet.A1_r / R_earth, planet.A1_P)
    ax[0, 1].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0, 1].set_ylabel(r"Pressure, $P$ [Pa]")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlim(0, None)
    
    ax[1, 1].plot(planet.A1_r / R_earth, planet.A1_T)
    ax[1, 1].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[1, 1].set_ylabel(r"Temperature, $T$ [K]")
    ax[1, 1].set_xlim(0, None)
    ax[1, 1].set_ylim(0, None)
    
    plt.tight_layout()
    plt.show()




sub_neptune = woma.Planet(
    A1_mat_layer    = ["ANEOS_Fe85Si15", "ANEOS_forsterite", "AQUA"],
    A1_T_rho_type   = ["adiabatic", "adiabatic", "adiabatic"],
    P_s             = 1e5,
    T_s             = 400,
    A1_R_layer      = [0.5 * R_earth, 0.9 * R_earth, 2.3 * R_earth],
)

sub_neptune.gen_prof_L3_find_M_given_R_R1_R2(M_max=10 * M_earth)

# sub_neptune.gen_prof_given_inner_prof(
#     mat         = "HM80_HHe", 
#     T_rho_type  = "adiabatic", 
#     rho_min     = 1, 
#     P_min       = 0,
# )

plot_spherical_profiles(sub_neptune)