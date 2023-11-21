import numpy as np


def calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh):
    #Return the Hill sphere radius (R_Hill) for an array of prograde BH where
    # R_Hill=a(q/3)^1/3 where a=semi-major axis, q=m_bh/M_SMBH
    
    bh_smbh_mass_ratio = prograde_bh_masses/(3.0*mass_smbh)
    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
    bh_hill_sphere = prograde_bh_locations*mass_ratio_factor
    #Return the BH Hill sphere radii for all orbiters. Prograde should have much larger Hill sphere
    return bh_hill_sphere

def encounter_test(prograde_bh_locations, bh_hill_sphere):
    #Using Hill sphere size and BH locations see if there are encounters within the Hill sphere
    # return indices of BH involved.

    # First sort the prograde bh locations in order from inner disk to outer disk
    sorted_bh_locations = np.sort(prograde_bh_locations)
    #Returns the indices of the original array in order, to get the sorted array
    sorted_bh_location_indices = np.argsort(prograde_bh_locations)
   
    #Find the appropriate (sorted) Hill sphere radii
    sorted_hill_spheres = bh_hill_sphere[sorted_bh_location_indices]

    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    separations = np.diff(sorted_bh_locations)

    # Note that separations are -1 of length of bh_locations
    # Take 1st location off locations array
    sorted_bh_locations_minus_first_element = sorted_bh_locations[1:len(sorted_bh_locations)]
    #Take last location off locations array
    sorted_bh_locations_minus_last_element = sorted_bh_locations[0:len(sorted_bh_locations)-1]

    # Separations are -1 of length of Hill_sphere array
    # Take 1st Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_first = sorted_hill_spheres[1:len(sorted_hill_spheres)]
    # Take last Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_last = sorted_hill_spheres[0:len(sorted_hill_spheres)-1]

    # Compare the Hill sphere distance for each BH with separation to neighbor BH
    #so we compare e.g. r2-r1 vs R_H1, r3-r2 vs R_H2
    comparison_distance_inwards = separations-sorted_hill_spheres_minus_last
    # and e.g. compare r2-r1 vs R_H2, r3-r2 vs R_H3
    comparison_distance_outwards = separations-sorted_hill_spheres_minus_first

    index_in = np.where(comparison_distance_inwards < 0)
    if isinstance(index_in,tuple):
        index_in = index_in[0]
    # E.g say r3-r2 <R_H2 then we'll want the info for the BH at r2 & r3. (i,i+1)
    index_out = np.where(comparison_distance_outwards < 0)
    if isinstance(index_out,tuple):
        index_out = index_out[0]
    #E.g. say r3-r2 <R_H3 then we'll want the info for the BH at r3 and r2 (i,i-1)
    length_index_in = len(index_in)
    length_index_out = len(index_out)

    new_indx_in = list(range(2*len(index_in)))
    new_indx_out = list(range(2*len(index_out)))
    #new_indx_in_bin_array = np.array[2*len(index_in),2]
    #temp_new_bin_array = np.ndarray[2*length_index_in,2]

    for ind in range(length_index_in):
        temp_index = index_in[ind]
        new_indx_in[2*ind] = temp_index
        new_indx_in[(2*ind)+1] = temp_index+1
    #    temp_new_bin_array = 
    print("new_indx_in",new_indx_in)
    temp_new_bin_in_array=np.reshape(new_indx_in,(len(index_in),2))
    print("ordered as bins",temp_new_bin_in_array)
    # Dynamics Here! Potential Double-binary or triple interaction!
    # For now! 
    # 0. Construct array of pairs of indices.
    # 1. Select binaries based on distance
    #     For binaries [i-1,i], [i,i+1] compare distance r(i)-r(i-1) to r(i+1)-r(i) and select smallest.
    #     Remove larger distane pair. E.g. if [i+1,i] is smaller, remove [i-1,i]
    #    But want to calculate 
    # 2. Fractional R_Hill for [i-1,i] vs [i,i+1]. Smaller fractional Hill radius wins
    # 3. TO DO Write a module for DiLaurentii+22 or Rowan+22 or LANL+22 phase space encounter and apply to all encounters 
    #          over timestep (10kyrs; assume random phase & number of encounters during timestep; pick randomly 
    #          from phase plots.) Also look at LANL group papers on binding energy of encounter.
    # 4. Ideally, consider the triple-dynamics encounter 
    
    # Search for repeat indices in binary array 
    unique_element,unique_index,unique_ct = np.unique(temp_new_bin_in_array,return_inverse = True,return_counts = True)
    print("unique elements",unique_element)
    print("unique index",unique_index)
    print("unique_count",unique_ct)

    repeats = unique_ct > 1
    repeat_indices = unique_element[repeats]
    print("repeat_indices",repeat_indices)
    print("separations",separations)
    print("sorted Hill spheres -last (Rh1,Rh2,..)",sorted_hill_spheres_minus_last)
    print("difference:")
    print("separations inward",comparison_distance_inwards)
    print("separations outward",comparison_distance_outwards)
    print("separations inward (repeat_indices)",comparison_distance_inwards[repeat_indices])
    print("separations outward (repeat indices)",comparison_distance_outwards[repeat_indices])
    #Compare nearest neighbour separations for BH in repeat_indices
    # E.g. separations =[r2-r1,r3-r2,r3-4]. If BH at r2 repeats in binary array 
    # e.g. bin_array=[[r1,r2] [r2,r3]..] then compare separations[r2-r1] to separations[r3-r2]
    # If r2-r1 < r3-r2 then make [r1,r2] the binary and remove [r2,r3]
    
    #if separations[repeat_indices] < separations[repeat_indices + 1]:
    #    temp_new_bin_in_array = np.delete(temp_new_bin_in_array,repeat_indices + 1)
    #    print("temp_bin_array",temp_new_bin_in_array)

#    for j in range(2*len(index_in)):    
#        temp_bin_in = new_indx_in_bin_array[temp_index[]]
   

#    for j in range(2*len(index_in)):    
#        temp_bin_in = new_indx_in_bin_array[temp_index[]]
   
    for ind in range(length_index_out):
        temp_index = index_out[ind]
        new_indx_out[2*ind] = temp_index
        new_indx_out[(2*ind)+1] = temp_index+1

    print("new_indx_out",new_indx_out)    
    temp_new_bin_out_array=np.reshape(new_indx_out,(len(index_out),2))
    print("ordered as bins",temp_new_bin_out_array)

    new_indxs = new_indx_in+new_indx_out
    #rindx = np.sort(new_indxs)
    result = np.asarray(new_indx_in)
    sorted_in_result = np.sort(result)

    new_result = np.asarray(new_indx_out)
    sorted_out_result = np.sort(new_result)
    print("sorted in result",sorted_in_result)
    print("sorted out result",sorted_out_result)
    # Concatenate the two lists, and remove duplicates
    final_bin_indices = np.array(list(set(list(sorted_in_result) + list(sorted_out_result))))
    sorted_final_bin_indices = np.sort(final_bin_indices)
    print("total final bin indices",sorted_final_bin_indices)
    #print("check if sorted_in & sorted_out arrays are the same")
    check = np.array_equiv(sorted_in_result, sorted_out_result)
    #print(check)
    # Return the indices of those elements in separation array <0
    # (ie BH are closer than 1 R_Hill)
    # In inwards case, r_i+1 -r_i <R_H_i, so relevant BH indices are i,i+1
   
    # In outwards case, r_i - r_i-1 <R_H_i so relevant BH indices are i,i-1
    final_1d_indx_array = sorted_in_result.flatten()
    sorted_final_1d_indx_array = np.sort(final_1d_indx_array)

    if len(sorted_final_1d_indx_array) > 0:
         print("Binary", sorted_final_1d_indx_array)

    return sorted_final_1d_indx_array

def binary_check(prograde_bh_locations, prograde_bh_masses, mass_smbh):
    """Determines which prograde BH will form binaries in this timestep. Takes as inputs
    the singleton BH locations & masses, and checks if their separations are less than
    the mutual Hill sphere of any 2 adjacent BH. If this is the case, determine the
    smallest separation pairs (in units of their mutual Hill sphere) to form a set of
    actual binaries (this module does handle cases where 3 or more bodies *might* form
    some set of binaries which would be mutually exclusive; however it does not handle
    or even flag the implied triple system dynamics). Returns an Nx2 array of the relevant
    binary indices, for further handling to form actual binaries & assign additional
    parameters (e.g. angular momentum of the binary).

    Parameters
    ----------
    prograde_bh_locations : float array
        locations of prograde singleton BH at start of timestep in units of 
        gravitational radii (r_g=GM_SMBH/c^2)
    prograde_bh_masses : float array
        initial masses of bh in prograde orbits around SMBH in units of solar masses
    mass_smbh : float
        mass of supermassive black hole in units of solar masses

    Returns
    -------
    all_binary_indices : [2,N] int array
        array of indices corresponding to locations in prograde_bh_locations, prograde_bh_masses,
        prograde_bh_spins, prograde_bh_spin_angles, and prograde_bh_generations which corresponds
        to binaries that form in this timestep. it has a length of the number of binaries to form (N)
        and a width of 2.
    """

    # First sort the prograde bh locations in order from inner disk to outer disk
    sorted_bh_locations = np.sort(prograde_bh_locations)
    # Returns the indices of the original array in order, to get the sorted array
    sorted_bh_location_indices = np.argsort(prograde_bh_locations)
    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    # Note length of separations is 1 less than prograde_bh_locations
    separations = np.diff(sorted_bh_locations)
    # Now compute mutual hill spheres of all possible binaries
    # same length as separations
    R_Hill_possible_binaries = (sorted_bh_locations[:-1] + separations/2.0) * \
        pow(((prograde_bh_masses[sorted_bh_location_indices[:-1]] + \
              prograde_bh_masses[sorted_bh_location_indices[1:]]) / \
                (mass_smbh * 3.0)), (1.0/3.0))
    # compare separations to mutual Hill spheres - negative values mean possible binary formation
    minimum_formation_criteria = separations - R_Hill_possible_binaries
    # collect indices of possible real binaries (where separation is less than mutual Hill sphere)
    index_formation_criteria = np.where(minimum_formation_criteria < 0)
    print(index_formation_criteria)
    # check for obviously independent binaries that are OK to form without further checks
    # versus sequences (ie mutually exclusive but temporarily repeated 'binaries')
    #check_seq = np.diff(index_formation_criteria)
    #print(check_seq)
    #binary_index_definite = np.extract(check_seq > 1, index_formation_criteria)
    #binary_index_check = np.extract(check_seq == 1, index_formation_criteria)
    #print(binary_index_definite)
    #print(binary_index_check)
    #actually I think that's too much work--the below algo should find all the binaries anyway...

    # Now deal with sequences: compute separation/R_Hill for all
    sequences_to_test = (separations[index_formation_criteria])/(R_Hill_possible_binaries[index_formation_criteria])
    print(sequences_to_test)
    # sort sep/R_Hill for all 'binaries' that need checking & store indices
    sorted_sequences = np.sort(sequences_to_test)
    print(sorted_sequences)
    sorted_sequences_indices = np.argsort(sequences_to_test)
    print(sorted_sequences_indices)
    # the smallest sep/R_Hill should always form a binary, so
    checked_binary_index = np.array([sorted_sequences_indices[0]])
    print(checked_binary_index)
    for i in range(len(sorted_sequences)):
        # if we haven't already counted it
        if (sorted_sequences_indices[i] not in checked_binary_index):
            # and it isn't the implicit partner of something we've already counted
            if (sorted_sequences_indices[i] not in checked_binary_index+1):
                # and the implicit partner of this thing isn't already counted
                if (sorted_sequences_indices[i]+1 not in checked_binary_index):
                    # and the implicit partner of this thing isn't already an implicit partner we've counted
                    if (sorted_sequences_indices[i]+1 not in checked_binary_index+1):
                        # then you can count it as a real binary
                        checked_binary_index = np.append(checked_binary_index, sorted_sequences_indices[i])

    print("THIS IS SAAVIK'S OUTPUT!!")
    # create array of all real binaries
    # BUT what are we returning? need indices of original arrays
    # that go to singleton vs binary assignments--actual binary formation *should* happen elsewhere!
    # I have the indices of the sorted_bh_locations array that correspond to actual binaries
    # these are checked_binary_index, checked_binary_index+1
    all_binary_indices = np.array([sorted_bh_location_indices[checked_binary_index], sorted_bh_location_indices[checked_binary_index+1]])
    print(np.shape(all_binary_indices))

    return all_binary_indices
