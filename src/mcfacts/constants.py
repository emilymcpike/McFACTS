"""
Defines a collection of global variables used throughout McFACTS.


Global Variables:
- mass_per_msun (float): number of Kg per Msun

"""

from astropy import constants as const

# mass_per_msun = 1.99e30


def rg_in_meters(smbh_mass):
    smbh_mass_units = smbh_mass * const.M_sun
    rg = (const.G * (smbh_mass_units) / (const.c ** 2.0)).to("meter")

    return (rg.value)
