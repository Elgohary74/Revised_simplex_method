import numpy as np
from Algorithm.revised_simplex import revised_simplex


def two_phase_simplex(A, b, c, tolerance=1e-10, max_iterations=1000):

    num_constraints, num_variables = A.shape

    # Phase 1
    A_phase1 = np.hstack([A, np.eye(num_constraints)])
    cost_phase1 = np.hstack([np.zeros(num_variables), np.ones(num_constraints)])
    try:
        x_phase1, z_phase1 = revised_simplex(A_phase1, b, cost_phase1, tolerance, max_iterations)
    except Exception:
        raise Exception("phase I failed so the Problem is infeasible")

    if z_phase1 > tolerance:
        raise Exception("original problem is infeasible")

    basic_indices_phase2 = [ # remove artificial variables from basis
        i for i, val in enumerate(x_phase1[:num_variables]) if val > tolerance
    ]

    while len(basic_indices_phase2) < num_constraints:
        for j in range(num_variables):
            if j not in basic_indices_phase2:
                basic_indices_phase2.append(j)
                if len(basic_indices_phase2) == num_constraints:
                    break

    nonbasic_indices_phase2 = [i for i in range(num_variables) if i not in basic_indices_phase2]
    basis_matrix = A[:, basic_indices_phase2]
    nonbasis_matrix = A[:, nonbasic_indices_phase2]
    basic_solution = np.linalg.solve(basis_matrix, b)
    nonbasic_solution = np.zeros(len(nonbasic_indices_phase2))

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        cost_basis = c[basic_indices_phase2]
        cost_nonbasis = c[nonbasic_indices_phase2]
        dual_multipliers = np.linalg.solve(basis_matrix.T, cost_basis)
        reduced_costs = cost_nonbasis - nonbasis_matrix.T @ dual_multipliers

        if all(reduced_costs >= -tolerance):
            x_optimal = np.zeros(num_variables)
            x_optimal[basic_indices_phase2] = basic_solution
            optimal_value = c @ x_optimal
            return x_optimal, optimal_value

        entering_index_in_nonbasis = np.argmin(reduced_costs)
        entering_variable = nonbasic_indices_phase2[entering_index_in_nonbasis]
        column_entering = A[:, entering_variable]

        direction_vector = np.linalg.solve(basis_matrix, column_entering)

        if all(direction_vector <= tolerance):
            raise Exception("LP is unbounded")

        ratios = np.array([
            basic_solution[i] / direction_vector[i] if direction_vector[i] > tolerance else np.inf
            for i in range(num_constraints)
        ])
        leaving_index_in_basis = np.argmin(ratios)
        leaving_variable = basic_indices_phase2[leaving_index_in_basis]
        step_size = ratios[leaving_index_in_basis]

        basic_indices_phase2[leaving_index_in_basis] = entering_variable
        nonbasic_indices_phase2[entering_index_in_nonbasis] = leaving_variable

        basis_matrix = A[:, basic_indices_phase2]
        nonbasis_matrix = A[:, nonbasic_indices_phase2]
        basic_solution = basic_solution - step_size * direction_vector
        basic_solution[leaving_index_in_basis] = step_size
        nonbasic_solution = np.zeros(len(nonbasic_indices_phase2))

    raise Exception("Maximum iterations reached")

