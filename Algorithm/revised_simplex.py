import numpy as np

def revised_simplex(A,b,c,tolerance=1e-10, max_iterations=1000): 
    
    num_constraints, num_variables = A.shape 
    basic_indices = list(range(num_variables - num_constraints, num_variables))
    nonbasic_indices = list(range(num_variables - num_constraints)) 
    
    basis_matrix = A[:, basic_indices]
    nonbasis_matrix = A[:, nonbasic_indices] 
    
    basic_solution = np.linalg.solve(basis_matrix, b)
    nonbasic_solution = np.zeros(len(nonbasic_indices)) 
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1  
        
        cost_basis = c[basic_indices]
        cost_nonbasis = c[nonbasic_indices]
        dual_multipliers = np.linalg.solve(basis_matrix.T, cost_basis)
        reduced_costs = cost_nonbasis - nonbasis_matrix.T @ dual_multipliers  
        
        if all(reduced_costs >= -tolerance):
            x_optimal = np.zeros(num_variables) 
            x_optimal[basic_indices] = basic_solution 
            optimal_value = c @ x_optimal 
            return x_optimal, optimal_value 
        
        #choose entering variable (most negative reduced cost) 
        entering_index_in_nonbasis = np.argmin(reduced_costs)
        entering_variable = nonbasic_indices[entering_index_in_nonbasis]
        column_entering = A[:, entering_variable]  
        direction_vector = np.linalg.solve(basis_matrix, column_entering) 
        
        if all(direction_vector <= tolerance):
            raise Exception("LP is unbounded")  
        
        #ratio test to determine leaving variable 
        ratios = np.array([
            basic_solution[i] / direction_vector[i] if direction_vector[i] > tolerance else np.inf
            for i in range(num_constraints)
        ])
        leaving_index_in_basis = np.argmin(ratios)
        leaving_variable = basic_indices[leaving_index_in_basis] 
        step_size = ratios[leaving_index_in_basis] #minimum ratio 
        
        #update basic and non-basic solutions
        basic_indices[leaving_index_in_basis] = entering_variable
        nonbasic_indices[entering_index_in_nonbasis] = leaving_variable  
        
        #update basis, non-basis matrices and solution
        basis_matrix = A[:, basic_indices]
        nonbasis_matrix = A[:, nonbasic_indices] 
        
        basic_solution = basic_solution - step_size * direction_vector  #update basic solution
        
        basic_solution[leaving_index_in_basis] = step_size
        nonbasic_solution = np.zeros(len(nonbasic_indices)) 
        
    raise Exception("maximum iterations reached")
        