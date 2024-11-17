from seagen import GenSphere
import numpy as np
import matplotlib.pyplot as plt


def make_particle_planet(r_profile, rho_pr0file, mat_profile, T_profile, n_particles):

    particles = GenSphere(
        n_particles,
        r_profile,
        rho_pr0file,
        A1_mat_prof=mat_profile,
        A1_T_prof=T_profile,
        verbosity=2,
    )


def test_gen_sphere_layers():
    """Generate spherical particle positions from a density profile with
    multiple layers, density discontinuities, and a temperature profile.

    Save a figure of the particles on the radial density and temperature
    profiles.
    """
    print(
        "\n==============================================================="
        "\n SEAGen sphere particles generation with a multi-layer profile "
        "\n==============================================================="
    )

    N_picle = 1e4

    # Profiles
    N_prof = int(1e6)
    A1_r_prof = np.arange(1, N_prof + 1) * 1 / N_prof
    # A density profile with three layers of different materials
    A1_rho_prof = 3 - 2 * A1_r_prof**2
    A1_rho_prof *= np.array(
        [1] * int(N_prof / 4) + [0.7] * int(N_prof / 2) + [0.3] * int(N_prof / 4)
    )
    A1_mat_prof = np.array(
        [0] * int(N_prof / 4) + [1] * int(N_prof / 2) + [2] * int(N_prof / 4)
    )
    A1_T_prof = 500 - 200 * A1_r_prof**2

    # Generate particles
    particles = GenSphere(
        N_picle,
        A1_r_prof,
        A1_rho_prof,
        A1_mat_prof=A1_mat_prof,
        A1_T_prof=A1_T_prof,
        verbosity=2,
    )

    # Figure
    plt.figure(figsize=(7, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(A1_r_prof, A1_rho_prof, c="b")
    ax1.scatter(particles.A1_r, particles.A1_rho, c="b")

    ax2.plot(A1_r_prof, A1_T_prof, c="r")
    ax2.scatter(particles.A1_r, particles.A1_T, c="r")

    ax1.set_xlabel("Radius")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("Temperature")
    ax1.yaxis.label.set_color("b")
    ax2.yaxis.label.set_color("r")

    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)

    plt.title("SEAGen Sphere Particles (Multi-Layer Profile)")

    plt.tight_layout()

    plt.show()


test_gen_sphere_layers()
