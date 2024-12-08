def tensor_product(v1, v2):
    """
    Compute the tensor product of two binary vectors.

    Parameters:
    v1 (list): The first binary vector.
    v2 (list): The second binary vector.

    Returns:
    list: The tensor product of the two binary vectors.
    """
    def simplify_product(term1, term2):
        """Simplify the product of two terms by removing duplicates."""
        combined = set(term1) | set(term2)  # Combine terms and remove duplicates
        return ''.join(sorted(combined))    # Return the sorted string representation

    result = []
    for elem1 in v1:
        for elem2 in v2:
            if elem1 == '0' or elem2 == '0':
                result.append('0')
            elif elem1 == '1':
                result.append(elem2)
            elif elem2 == '1':
                result.append(elem1)
            else:
                result.append(simplify_product(elem1, elem2))
    return result

def generate_pprm_expansions(n):
    """
    Generate PPRM expansions for n binary variables.

    Parameters:
    n (int): Number of binary variables.

    Returns:
    list: The PPRM expansions in lexicographical order.
    """
    # Initialize with the first variable
    expansions = ['1', chr(65 + n - 1)]  # Start with ['1', 'C'] for n=3

    # Iterate through remaining variables
    for i in range(n - 2, -1, -1):  # Start from 'B' (65 + n - 2) down to 'A'
        current_var = chr(65 + i)
        expansions = tensor_product(['1', current_var], expansions)

    return expansions

if __name__ == "__main__":
    # Example usage:
    n = 3  # Number of binary variables
    pprm_expansions = generate_pprm_expansions(n)
    print(pprm_expansions)
