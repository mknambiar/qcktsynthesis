from itertools import combinations
from tenpprm import generate_pprm_expansions
import logging

def sub_xor_combined(pprm_vector, n, var_index, factors):
    """
    Modify the PPRM table based on the combined sub_xor_1 and sub_xor_factor algorithms.

    Parameters:
    pprm_vector (list): The PPRM expansion as a list of 0s and 1s.
    n (int): Number of binary variables.
    var_index (int): Index of the variable (1-based, e.g., 1 for 'A', 2 for 'B').
    factors (list): List of binary variable indexes similar to var_index.

    Returns:
    list: Modified PPRM vector.
    """
    # Generate PPRM terms
    logging.debug("n = %s", n)
    pprm_terms = generate_pprm_expansions(n)
    logging.debug("input pprm_terms = %s", pprm_vector)
    
    # Get the variable based on the index
    #variable = chr(64 + var_index)  # chr(64 + 1) is 'A', chr(64 + 2) is 'B', etc.
    variable = chr(64 + n - var_index)
    logging.debug("var_index = %d, variable = %s", var_index, variable) 

    # Find indices of terms that contain the variable and have an entry of 1 in the PPRM vector
    term_indices_with_var = [i for i, term in enumerate(pprm_terms) if variable in term and pprm_vector[i] == 1]
    logging.debug("term_indices_with_var = %s", term_indices_with_var)

    # Dictionary to count occurrences of new indices
    new_index_count = {}

    for term_index in term_indices_with_var:
        # Remove the variable from the term
        term_without_var = pprm_terms[term_index].replace(variable, '')
        
        # Add symbolic boolean multiplication with the variables in factors if factors is not [0] or [0]
        if factors == []:
            new_term = term_without_var
        else:
            term_factors = ''.join(chr(64 + n - factor) for factor in factors)  # Convert factors to variable names
            new_term = ''.join(sorted(set(term_without_var + term_factors)))  # Ensure no duplicates and sorted

        # Find the index of the new term in the PPRM terms
        #logging.debug("new term = ", new_term, "var_index = ", var_index, "factors = ", factors, "term_index = ", term_index)
        logging.debug("new term = %s , var_index = %d , factors = %s, term_index = %d ", new_term, var_index, factors, term_index)
        if (new_term != ''):
            new_index = pprm_terms.index(new_term)
        else:
            new_index = 0  # The empty term corresponds to the index 0

        # Increment the count of the new index
        if new_index in new_index_count:
            new_index_count[new_index] += 1
        else:
            new_index_count[new_index] = 1 + pprm_vector[new_index]


    # Modify the PPRM vector based on the count
    for index, count in new_index_count.items():
        if count % 2 == 0:
            pprm_vector[index] = 0
        else:
            pprm_vector[index] = 1
            
    logging.debug("output pprm_terms = %s", pprm_vector)
    return pprm_vector

# Example usage:
if __name__ == "__main__":
    pprm_vector = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]  # Example PPRM vector
    n = 4  # Number of binary variables
    var_index = 1  # Index of variable 'A'
    factors = [2, 3]  # Additional factors (e.g., 'B' and 'C')
    modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factors)
    print(modified_pprm_vector)
    pprm_vector = [0, 0, 1, 0, 1, 1, 0, 0]  # Example PPRM vector
    n = 3  # Number of binary variables
    var_index = 2  # Index of variable 'A'
    factors = [1, 3]  # Additional factors (e.g., 'B' and 'C')
    modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factors)
    print(modified_pprm_vector)
    n = 3
    var_index = 3  # Index of variable 'A'
    factors = [1, 2]  # Additional factors (e.g., 'B' and 'C')
    modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factors)
    print(modified_pprm_vector)
    pprm_vector = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]  # Example PPRM vector
    n = 4  # Number of binary variables
    var_index = 1  # Index of variable 'A'
    factors = [0]  # Additional factors (e.g., 'B' and 'C')
    modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factors)
    print(modified_pprm_vector)
    pprm_vector = [0, 0, 1, 0, 1, 1, 0, 0]  # Example PPRM vector
    n = 3  # Number of binary variables
    var_index = 1  # Index of variable 'A'
    factors = [0] #subxor1
    modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factors)
    print(modified_pprm_vector)
    n=3
    pprm_vector = [0, 0, 1, 0, 1, 1, 0, 0]  # Example PPRM vector
    var_index = 2  # Index of variable 'B'
    factors = [0] #subxor1
    modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factors)
    print(modified_pprm_vector)