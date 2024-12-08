from queue import PriorityQueue
import time
import logging
import sys
import numpy as np
import copy
from subxorc import sub_xor_combined 

class PPRM:
    
    def __init__(self, a):
        '''
        a: input vector, indicating the coefficients of f 
        '''
        self.a = a.astype(np.uint8)
        self.k = np.log2(self.a.shape[0]).astype(np.uint8)
        self.b = self.Binary_PPRM(self.a, self.k)
        self.X = self.X(self.k)

    
    def Binary_PPRM(self, a, k):
        blocksize = 1
        logging.debug("input a = %s", a)
        for i in range(k):
            mask = np.array([[1]*(2**i), [0]*(2**i)]*(2**(k-(i+1)))).reshape(2**k,1).astype(np.uint8)
            temp = np.zeros_like(a)
            # temp = a_masked SHR(shift right) blocksize
            temp[blocksize:] = (a & mask)[:-blocksize]
            # XOR between all blocks
            a = a ^ temp; 
            blocksize *= 2
        logging.debug("output a = %s", a)
        return a


    def x_product(self, x_i, x_j):
        '''
        x_i: [i],  x_j: [j]
        x_product([i], [j]) = [i, j]   : x_i . x_j
        x_product(x_i, 1) = [i]        : x_i . 1 = x_i

        For example:
           rm_concat([2], 1) = [2]            : x2 . 1 = x2
           rm_concat([2], [0]) = [0, 2]       : x0 . x2 
           rm_concat([0, 1], [2]) = [0, 1, 2] : x0 . x1 . x2
       '''
        if x_i == 1:
            return x_j
        if x_j == 1:
            return x_i
        return list(np.concatenate((x_j, x_i)))
    

    def x_kron(self, list2, list1):
        ''' kronecker product of lists of product terms
            x_kron([1,[1]], [1,[0]]) = [1, [0], [1], [0, 1]]
        '''
        output = list1
        for i in range(len(list1)):
            output.append(self.x_product(list2[1], list1[i]))
        return output
    

    def X(self, k):
        # time0 = time.perf_counter_ns()
        '''
        X(1) = [1, [0]]               : 1 xor x0

        X(2) = [1, [0], [1], [0, 1]]  : 1 xor x0 xor x1 xor x0.x1
                
        X(3) = [1, [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]] 
        '''
        if k == 1:
            return [1, [0]]
        if k == 2:
            return self.x_kron([1, [1]], [1, [0]])

        result = [1, [0]]  # Start with the result for k = 1
        for i in range(2, k+1):
            result = self.x_kron([1, [i-1]], result)
            
        # time1= time.perf_counter_ns() - time0
        # print("time1:", time1/(10**9), " seconds")
        return result


    
    def pprm(self):
        ''' Returns a list indicating which product terms are required
        for the new PPRM expression.

        For example:

            b = array([[0],
                       [1],
                       [1],
                       [0],
                       [1],
                       [0],
                       [0],
                       [0]])

            pprm() = [[0], [1], [2]]  : x0 xor x1 xor x2
        '''
        result = []
        idx_ones_in_b = np.argwhere(self.b == 1)[:, 0].reshape(-1)
        for idx in idx_ones_in_b:
            result.append(self.X[idx])
        return result

# Usage
# a = np.array([[0],
              # [1],
              # [1],
              # [0],
              # [1],
              # [0],
              # [0],
              # [1]])
# k=3
# a = np.random.randint(0, 2, size=(2**k, 1))

# pprm_instance = PPRM(a)
# result = pprm_instance.pprm()

class Timer:
    def __init__(self, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

    def is_expired(self):
        return (time.time() - self.start_time) >= self.time_limit

class Node:
    def __init__(self, depth, parent=None, factor=None, pprm=None, variable=None, gate_factor = None, terms=0, elim=0, priority=float('inf')):
        self.depth = depth
        self.parent = parent
        self.factor = factor
        self.pprm = pprm
        self.gate_variable = variable
        self.gate_factor = gate_factor
        self.terms = terms
        self.elim = elim
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority
        
    def print(self):
        if (self.gate_variable != None):
            print("")
            print("gate operation : ", self.gate_variable, " = ", self.gate_variable, " xor ", self.gate_factor)
            print("")
        print("=======Qubit State=======")
        for i in range(len(self.pprm)):
            print(self.pprm[i].pprm())

def generate_pprm(function):
    # Dummy PPRM generation for the provided function
    result = []
    for i in range(len(function)):
        logging.debug("function %d = %s",i, function[i])
        a = np.array(function[i]).reshape(-1, 1)
        pprm_instance = PPRM(a)
        result.append(pprm_instance)
        logging.debug("result = %s", pprm_instance.b)
        logging.debug("indexes = %s", pprm_instance.pprm())
    return result  # Replace with actual PPRM generation logic

def expand_pprm(pprm, variable):
    # Dummy expansion of the PPRM
    pprm_instance = PPRM(a)
    result = pprm_instance.pprm()
    return [factor for factor in pprm if variable not in factor]

def substitute_pprm(pprm, variable, factor):
    # Dummy substitution in the PPRM
    logging.debug("input factor = %s", factor)
    result = []
    #if (len(factor) == 0):
        #factor = [0]
    for i in range(len(pprm)):
        pprm_instance = pprm[i]
        pprm_vector = [item for sublist in pprm_instance.b for item in sublist]       
        logging.debug("pprm_instance expansion = %s", pprm_vector)
        n = pprm_instance.k
        var_index = variable #this part needs to be double checked
        logging.debug("factor = %s", factor)
        modified_pprm_vector = sub_xor_combined(pprm_vector, n, var_index, factor)
        new_pprm_instance = copy.copy(pprm_instance)
        new_pprm_instance.b = np.array(modified_pprm_vector).reshape(-1, 1) #Shallow copy looks like the right thing. No need to update anything else
        result.append(new_pprm_instance)
        
    return result
    
def count_terms(pprm):
    
    terms = 0
    for i in range(len(pprm)):
        pprm_instance = pprm[i]
        var_list = pprm_instance.pprm()
        terms = terms + len(var_list)

    return terms
    
    
def find_variables(pprm):
    variable_set = set()
    for i in range(len(pprm)):
        pprm_instance = pprm[i]
        var_list = pprm_instance.pprm()
        for j in range(len(var_list)):
            element = var_list[j]
            if (element == 1):
                continue
            logging.info("element = %s",element)
            for k in range(len(element)):
                logging.debug("k = %s", element[k])
                variable_set.add(element[k])
    return variable_set
    
def find_factors_in_pprm(pprm, variable):
    factor_list = []
    pprm_instance = pprm[variable]

    var_list = pprm_instance.pprm()
    for j in range(len(var_list)):
        element = var_list[j]
        #if (element == 1):
        #    continue
        logging.info("element = %s",element) 
        logging.info("variable = %d", variable)
        factor_list.append(element)

    return factor_list

def positive_polarity_reed_muller(function, time_limit=60):
    best_depth = float('inf')
    best_sol_node = None
    ideal_depth = len(function)
    #init_terms = function.count(1)
    #logging.debug("number of 1 terms in the function is = %s", init_terms)
    
    timer = Timer(time_limit)

    original_pprm = generate_pprm(function)
    # Initialize root node
    root_node = Node(depth=0, pprm = original_pprm)
    root_node.terms = count_terms(original_pprm)
    root_node.elim = root_node.terms - root_node.terms
    logging.debug("TT. Root node terms = %d", root_node.terms)

    pq = PriorityQueue()
    root_node.priority = float('inf')
    pq.put((root_node.priority, root_node))
    
    #input("Press Enter to continue...") 
    #print("The program has resumed.")

    while not pq.empty() and not timer.is_expired():
        parent_node = pq.get()[1]
        if parent_node.depth >= best_depth - 1:
            continue
            
        logging.debug("==============================Dqueued===========================================================")
        
        variable_list = find_variables(parent_node.pprm)
        logging.debug("variable_list = %s", variable_list)
        #for variable in parent_node.pprm:
        for variable in variable_list:
            #v_i_pprm = expand_pprm(parent_node.pprm, variable)
            v_i_pprm = find_factors_in_pprm(parent_node.pprm, variable)
            logging.debug("factor_list = %s variable =%d ---------------------------------------------------------------------", v_i_pprm, variable)
            for factor in v_i_pprm:
                if (factor == 1):
                    new_factor = []
                else:
                    new_factor = factor
                logging.debug("new_factor = %s ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", new_factor)
                if variable not in new_factor:
                    child_node = Node(depth=parent_node.depth + 1, parent=parent_node)
                    logging.debug("variable = %s", variable)
                    child_node.pprm = substitute_pprm(parent_node.pprm, variable, new_factor)
                    child_node.gate_variable = variable
                    child_node.gate_factor = new_factor
                    child_node.terms = count_terms(child_node.pprm)
                    child_node.elim = parent_node.terms - child_node.terms

                    if child_node.terms == ideal_depth and child_node.depth < best_depth:
                        best_depth = child_node.depth
                        best_sol_node = child_node

                    if child_node.elim > 0:
                        child_node.priority = child_node.depth + child_node.elim
                        pq.put((child_node.priority, child_node))
                        logging.debug("PP:Putting the child node ==============")
                        logging.debug("TT. Child node terms = %d, Parent Node terms = %d, ", child_node.terms, parent_node.terms)
                    else:
                        logging.debug("NP:Not Putting the child node =========")

    return best_sol_node

# Example usage
#function = ['v1', 'v2', 'v3', '...']  # Replace with the actual truth table or function representation
#function = ['0', '1', '1', '1']


# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,              # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    #format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log message format
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Send logs to standard output (stdout)
    ]
)

logging.getLogger().setLevel(logging.CRITICAL)

# Example log messages
logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("Testing: This is a critical message")



#function = [0, 1, 1, 0, 0, 1, 1, 0]

function = [[0, 0, 1, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
#there is an order here functions(truth tables) for  c,b, a are listed in that order because c = 0, b = 1, a = 2

print("")
print("Input function (Truth tables)")
for i in range(len(function)):
    print(function[i])


best_solution = positive_polarity_reed_muller(function, time_limit=60)

# Print the best solution
if best_solution:
    path = []
    node = best_solution
    path.append(node)
    while node:
        if node.parent:
            path.append(node.parent)
        node = node.parent
    path.reverse()
    
    
    print("")
    print("Best solution path:")
    
    for step in path:
        #print(step)
        step.print()
        
    print("")
    print("How to understand this output:")
    print("Qubits are numbered starting from 0 upwards:")
    print("In a Qubit state and the input function truth table there are as many rows as qubits")
    print("First row is for qubit 0, 2nd row for qubit 1 and so on")
    
else:
    print("No solution found within the time limit.")
