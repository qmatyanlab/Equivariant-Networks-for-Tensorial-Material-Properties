from e3nn.o3 import Irreps

def BalancedIrreps(lmax, vec_dim):
    """ Allocates irreps equally along channel budget, resulting
        in unequal numbers of irreps in ratios of 2l_i + 1 to 2l_j + 1.
        Uses full set of irreps with redundance

    Parameters
    ----------
    lmax : int
        Maximum order of irreps.
    vec_dim : int
        Dim of feature vector.

    Returns
    -------
    Irreps
        Resulting irreps for feature vectors.

    """
    irrep_spec = "0e"
    for l in range(1, lmax + 1):
            irrep_spec += " + {0}e + {0}o".format(l)
    irrep_spec_split = irrep_spec.split(" + ")
    dims = [int(irrep[0]) * 2 + 1 for irrep in irrep_spec_split]
    # Compute ratios
    ratios = [1 / dim for dim in dims]
    # Determine how many copies per irrep
    irrep_copies = [int(vec_dim * r / len(ratios)) for r in ratios]
    # Determine the current effective irrep sizes
    irrep_dims = [n * dim for (n, dim) in zip(irrep_copies, dims)]
    # Add trivial irreps until the desired size is reached
    irrep_copies[0] += vec_dim - sum(irrep_dims)

    # Convert to string
    str_out = ''
    for (spec, dim) in zip(irrep_spec_split, irrep_copies):
        str_out += str(dim) + 'x' + spec
        str_out += ' + '
    str_out = str_out[:-3]
    # Generate the irrep
    irreps_in1 = Irreps(str_out)
    print('Determined irrep type:', irreps_in1)
    return irreps_in1