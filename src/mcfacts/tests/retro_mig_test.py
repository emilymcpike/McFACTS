import numpy as np
import scipy

from mcfacts.physics.migration.type1 import retro_mig
from mcfacts.physics.eccentricity import retro_ecc
from mcfacts.physics.disk_capture import capture_inc_damp

if __name__ == "__main__":
    # I just want to run retro_mig.py on its own to see if it runs, so I'm
    #   setting up this dummy test
    # and while I'm at it, I'll test retro_ecc.py too... and eventually I bet
    #   I'll add retro_inc, or even just some kind of inc method...

    # But I need to import a surface density profile and set it up as a function
    infile = "../inputs/data/sirko_goodman_surface_density.txt" # this is violence, sorry
    surface_density_file = open(infile, 'r')
    density_list = []
    radius_list = []
    for line in surface_density_file:
        line = line.strip()
        # If it is NOT a comment line
        if (line.startswith('#') == 0):
            columns = line.split()
            #If radius is less than disk outer radius
            #if columns[1] < disk_outer_radius:
            density_list.append(float(columns[0]))
            radius_list.append(float(columns[1]))
    # close file
    surface_density_file.close()

    # re-cast from lists to arrays
    surface_density_array = np.array(density_list)
    disk_model_radius_array = np.array(radius_list)

    # create function
    surf_dens_func_log = scipy.interpolate.UnivariateSpline(
        disk_model_radius_array, np.log(surface_density_array))
    surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(x))

    mass_smbh = 1.0e8
    retrograde_bh_locations = np.array([1.e3, 1.e3, 1.e3, 1.e3, 1.e3, 1.e3, 1.e3])
    retrograde_bh_masses = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    retrograde_bh_orb_ecc = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    retrograde_bh_orb_inc = (np.pi - 1.e-6) * np.ones(7)
    retro_arg_periapse = 0.3 * np.pi * np.ones(7)
    timestep = 1e4

    thing1 = retro_mig.retro_mig(
        mass_smbh,
        retrograde_bh_locations,
        retrograde_bh_masses,
        retrograde_bh_orb_ecc,
        retrograde_bh_orb_inc,
        retro_arg_periapse,
        timestep,
        surf_dens_func)
    
    print("retro_mig")
    print(thing1)

    thing2 = retro_ecc.retro_ecc(
        mass_smbh,
        retrograde_bh_locations,
        retrograde_bh_masses,
        retrograde_bh_orb_ecc,
        retrograde_bh_orb_inc,
        retro_arg_periapse,
        timestep,
        surf_dens_func)
    
    print("retro_ecc")
    print(thing2)

    thing3 = capture_inc_damp.orb_inc_damping(
        mass_smbh,
        retrograde_bh_locations,
        retrograde_bh_masses,
        retrograde_bh_orb_ecc,
        retrograde_bh_orb_inc,
        retro_arg_periapse,
        timestep,
        surf_dens_func)
    
    print("retro_inc")
    print(thing3)