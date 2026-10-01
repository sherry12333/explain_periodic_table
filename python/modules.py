
#from pylab import * # needed for symbols in pyplot ::::: not with pylab inline     
import numpy as np #                        #::::: not with pylab inline
import scipy
#from scipy import linalg                   # ::::: not with pylab inline ??     
#from scipy import special  
#from scipy import stats  
from scipy.interpolate import InterpolatedUnivariateSpline 
import matplotlib.pyplot as plt   
import cmath  






def cal_ryo_E(eta, iteration, charge, epsilon0, N10, N20, N21, N30, N31, N32, N40, 
              xarray_P, xarray_V, tknot_P, tknot_V, kord, Gauss_mat, parameter1, 
              parameter2, l0, l1, l2, Z):
    # Hydrogen-like solution
    c0, D0, bhb_matrix0, BB_matrix0 = cal_eigen(xarray_P, tknot_P, kord, Gauss_mat, parameter1, parameter2, l0, Z)
    c1, D1, bhb_matrix1, BB_matrix1 = cal_eigen(xarray_P, tknot_P, kord, Gauss_mat, parameter1, parameter2, l1, Z)
    c2, D2, bhb_matrix2, BB_matrix1 = cal_eigen(xarray_P, tknot_P, kord, Gauss_mat, parameter1, parameter2, l2, Z)

    E_hyd1 = D0[0, 0]
    E_hyd2 = D0[1, 1]
    E_hyd3 = D0[2, 2]

    Bavx1, dBavx1, dB2avx1 = bsplgen(xarray_P, tknot_P, kord)

    P10 = np.dot(Bavx1[:, 1:-1], c0[:, 0])
    P20 = np.dot(Bavx1[:, 1:-1], c0[:, 1])
    P21 = np.dot(Bavx1[:, 1:-1], c1[:, 0])
    P30 = np.dot(Bavx1[:, 1:-1], c0[:, 2])
    P31 = np.dot(Bavx1[:, 1:-1], c1[:, 1])
    P32 = np.dot(Bavx1[:, 1:-1], c2[:, 0])
    P40 = np.dot(Bavx1[:, 1:-1], c0[:, 3])

    ryo_ne = (1 / (4 * np.pi)) * charge * (
        N10 * (P10 / xarray_P) ** 2 + N20 * (P20 / xarray_P) ** 2 + N21 * (P21 / xarray_P) ** 2 +
        N30 * (P30 / xarray_P) ** 2 + N31 * (P31 / xarray_P) ** 2 + N32 * (P32 / xarray_P) ** 2 +
        N40 * (P40 / xarray_P) ** 2
    )

    RHS_ne = -xarray_P * ryo_ne * 4 * np.pi / (4 * np.pi * epsilon0)
    RHS_ne[0] = RHS_ne[1]

    dB2_matrix = build_matrix_dB2(xarray_P, dB2avx1, dBavx1)
    c = l_u(dB2_matrix, RHS_ne)
    dB_end_end_1 = dBavx1[-1, -2]
    dB_end = dBavx1[-1, -1]
    c_end_1 = c[-1]

    c_end = (0 - dB_end_end_1) / dB_end * c_end_1
    c = np.concatenate([[0], c, [c_end]])

    # Electron potential is considered
    bhb_V_ee_matrix_old = cal_V_ee_matrix(c, bhb_matrix0, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, Gauss_mat, parameter1, parameter2, l0, Z)
    V_old = bhb_V_ee_matrix_old

    eigen10 = np.zeros(iteration)
    eigen20 = np.zeros(iteration)
    eigen21 = np.zeros(iteration)
    eigen30 = np.zeros(iteration)
    eigen31 = np.zeros(iteration)
    eigen32 = np.zeros(iteration)
    eigen40 = np.zeros(iteration)

    for i in range(iteration):
        c0, D0 = cal_eigen1(bhb_V_ee_matrix_old, bhb_matrix0, BB_matrix0)
        c1, D1 = cal_eigen1(bhb_V_ee_matrix_old, bhb_matrix1, BB_matrix1)
        c2, D2 = cal_eigen1(bhb_V_ee_matrix_old, bhb_matrix2, BB_matrix1)

        P10 = np.dot(Bavx1[:, 1:-1], c0[:, 0])
        P20 = np.dot(Bavx1[:, 1:-1], c0[:, 1])
        P21 = np.dot(Bavx1[:, 1:-1], c1[:, 0])
        P30 = np.dot(Bavx1[:, 1:-1], c0[:, 2])
        P31 = np.dot(Bavx1[:, 1:-1], c1[:, 1])
        P32 = np.dot(Bavx1[:, 1:-1], c2[:, 0])
        P40 = np.dot(Bavx1[:, 1:-1], c0[:, 3])

        ryo_ne = (1 / (4 * np.pi)) * charge * (
            N10 * (P10 / xarray_P) ** 2 + N20 * (P20 / xarray_P) ** 2 + N21 * (P21 / xarray_P) ** 2 +
            N30 * (P30 / xarray_P) ** 2 + N31 * (P31 / xarray_P) ** 2 + N32 * (P32 / xarray_P) ** 2 +
            N40 * (P40 / xarray_P) ** 2
        )

        RHS_ne = -xarray_P * ryo_ne * 4 * np.pi / (4 * np.pi * epsilon0)
        RHS_ne[0] = RHS_ne[1]

        dB2_matrix = build_matrix_dB2(xarray_P, dB2avx1, dBavx1)
        c = l_u(dB2_matrix, RHS_ne)
        dB_end_end_1 = dBavx1[-1, -2]
        dB_end = dBavx1[-1, -1]
        c_end_1 = c[-1]

        c_end = (0 - dB_end_end_1) / dB_end * c_end_1
        c = np.concatenate([[0], c, [c_end]])

        eigen10[i] = D0[0, 0]
        eigen20[i] = D0[1, 1]
        eigen21[i] = D1[0, 0]
        eigen30[i] = D0[2, 2]
        eigen31[i] = D1[1, 1]
        eigen32[i] = D2[0, 0]
        eigen40[i] = D0[3, 3]

        bhb_V_ee_matrix_new = cal_V_ee_matrix(c, bhb_matrix0, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, Gauss_mat, parameter1, parameter2, l0, Z)

        V_new = bhb_V_ee_matrix_new
        V_new = V_new * (1 - eta) + V_old * eta
        V_old = V_new
        bhb_V_ee_matrix_old = V_new

    ryo_ne[0] = 0

    size_c = c0.shape

    E_orb10 = eigen10[-1]
    E_orb20 = eigen20[-1]
    E_orb21 = eigen21[-1]
    E_orb30 = eigen30[-1]
    E_orb31 = eigen31[-1]
    E_orb32 = eigen32[-1]
    E_orb40 = eigen40[-1]

    sum10 = sum20 = sum21 = sum30 = sum31 = sum32 = sum40 = 0
    for i in range(size_c[0]):
        sum10 += np.dot(bhb_V_ee_matrix_new[i, :], c0[i, 0] * c0[:, 0])
        sum20 += np.dot(bhb_V_ee_matrix_new[i, :], c0[i, 1] * c0[:, 1])
        sum21 += np.dot(bhb_V_ee_matrix_new[i, :], c1[i, 0] * c1[:, 0])
        sum30 += np.dot(bhb_V_ee_matrix_new[i, :], c0[i, 2] * c0[:, 2])
        sum31 += np.dot(bhb_V_ee_matrix_new[i, :], c1[i, 1] * c1[:, 1])
        sum32 += np.dot(bhb_V_ee_matrix_new[i, :], c2[i, 0] * c2[:, 0])
        sum40 += np.dot(bhb_V_ee_matrix_new[i, :], c0[i, 3] * c0[:, 3])

    E_tot10 = N10 * (E_orb10 - sum10 / 2)
    E_tot20 = N20 * (E_orb20 - sum20 / 2)
    E_tot21 = N21 * (E_orb21 - sum21 / 2)
    E_tot30 = N30 * (E_orb30 - sum30 / 2)
    E_tot31 = N31 * (E_orb31 - sum31 / 2)
    E_tot32 = N32 * (E_orb32 - sum32 / 2)
    E_tot40 = N40 * (E_orb40 - sum40 / 2)

    E_tot = E_tot10 + E_tot20 + E_tot21 + E_tot30 + E_tot31 + E_tot32 + E_tot40
    return E_tot, ryo_ne


# calculate potential V_ee_dir at transform_coor

def cal_potential(transform_coor, xintervall, tknot, kord, RHS):
    # Generate B-spline basis and derivatives
    Bavx, dBavx, dB2avx = bsplgen(xintervall, tknot, kord)
    dB2_matrix = build_matrix_dB2(xintervall, dB2avx, dBavx)

    Bavx1, dBavx1, dB2avx1 = bsplgen([transform_coor, transform_coor + 1], tknot, kord)
    B_Gaussian = Bavx1[0, :]

    # Solve the linear system for coefficients
    c = l_u(dB2_matrix, RHS)
    dB_end_end_1 = dBavx[-1, -2]
    dB_end = dBavx[-1, -1]
    c_end_1 = c[-1]

    c_end = (0 - dB_end_end_1) / dB_end * c_end_1
    c0 = 0
    c = np.concatenate([[c0], c, [c_end]])

    # Calculate the potential
    result = np.dot(B_Gaussian, c)
    Pot = result / transform_coor

    return Pot
# the return value maybe Pot and result

# build matrix dB2

def build_matrix_dB2(xintervall, dB2, dB1):
    # Initialize a square matrix of zeros with size equal to the length of xintervall
    n = len(xintervall)
    matrix = np.zeros((n, n))

    # Fill the matrix according to the logic in the MATLAB code
    for i in range(n - 1):
        matrix[i, i] = dB2[i, i + 1]
        matrix[i, i + 1] = dB2[i, i + 2]
        matrix[i + 1, i] = dB2[i + 1, i + 1]

    # Handle the last element separately
    matrix[-1, -1] = dB2[-1, -2] - (dB1[-1, -2] * dB2[-1, -1]) / dB1[-1, -1]

    return matrix

# % build matrix B

def build_matrix_B(xintervall, dB2):
    # Initialize a square matrix of zeros with size equal to the length of xintervall
    n = len(xintervall)
    matrix = np.zeros((n, n))

    # Fill the matrix according to the MATLAB logic
    for i in range(n - 1):
        matrix[i, i] = dB2[i, i + 1]
        matrix[i, i + 1] = dB2[i, i + 2]
        matrix[i + 1, i] = dB2[i + 1, i + 1]

    # Handle the last element separately
    matrix[-1, -1] = dB2[-1, -2]

    return matrix


# % LU factorization
def l_u(A, B):
    # Convert A to a sparse matrix
    A_sparse = csc_matrix(A)

    # Perform LU factorization
    lu = splu(A_sparse)

    # Solve LY = B
    Y = lu.L.dot(np.linalg.solve(lu.L.A, B))

    # Solve UX = Y
    X = np.linalg.solve(lu.U.A, Y)

    return X

# insert extra x coordinate into array, the added x coordinate is not equal to items in


def insert_array(x, xarray):
    """
    Inserts a value x into a sorted array xarray at the correct position.
    The inserted value x should not already exist in the array.

    Parameters:
    x (float): The value to insert.
    xarray (list or np.ndarray): The sorted array into which x will be inserted.

    Returns:
    np.ndarray: A new array with x inserted.
    """
    # Ensure xarray is a NumPy array
    xarray = np.array(xarray)

    # Find the insertion index
    index = 0
    for i in range(len(xarray) - 1):
        if x > xarray[i] and x < xarray[i + 1]:
            index = i
            break

    # Insert x at the identified index
    xarray_insert = np.concatenate((xarray[:index + 1], [x], xarray[index + 1:]))

    return xarray_insert

# calculate P=rR

def cal_eigen(xarray, tknot, kord, Gauss_mat, parameter1, parameter2, l, Z):
    """
    Calculate eigenvalues and eigenvectors for the system.

    Parameters:
    xarray (np.ndarray): Array of x values.
    tknot (np.ndarray): Knot sequence.
    kord (int): Order of the B-spline.
    Gauss_mat (np.ndarray): Gaussian quadrature matrix.
    parameter1, parameter2 (float): Additional parameters for calculation.
    l (float): Angular momentum quantum number.
    Z (float): Nuclear charge.

    Returns:
    tuple: Eigenvectors (V), eigenvalues (D), bhb_matrix, BB_matrix.
    """
    # Initialize BB_matrix
    BB_matrix = np.zeros((len(tknot) - kord - 2, len(tknot) - kord - 2))

    for j in range(1, len(tknot) - kord - 1):  # MATLAB indices start at 1
        for i in range(1, len(tknot) - kord - 1):
            BB_matrix[j - 1, i - 1] = cal_inte_BBinterval(xarray, tknot, kord, i, j, Gauss_mat)

    # Initialize bhb_matrix
    bhb_matrix = np.zeros((len(tknot) - kord - 2, len(tknot) - kord - 2))

    for j in range(1, len(tknot) - kord - 1):
        for i in range(1, len(tknot) - kord - 1):
            bhb_matrix[j - 1, i - 1] = cal_bhb_interval(xarray, tknot, kord, i, j, parameter1, parameter2, Gauss_mat, l, Z)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(bhb_matrix, BB_matrix)

    return eigenvectors, eigenvalues, bhb_matrix, BB_matrix

# a little difference from the one above

def cal_eigen1(bhb_V_ee_matrix, bhb_matrix, BB_matrix):
    """
    Calculate eigenvalues and eigenvectors for a generalized eigenvalue problem.

    Parameters:
    bhb_V_ee_matrix (np.ndarray): Interaction matrix (e.g., from electron-electron interaction).
    bhb_matrix (np.ndarray): Original BHB matrix.
    BB_matrix (np.ndarray): Overlap matrix.

    Returns:
    tuple: Eigenvectors (V) and eigenvalues (D).
    """
    # Combine bhb_V_ee_matrix and bhb_matrix
    bhb_matrix1 = bhb_V_ee_matrix + bhb_matrix
    
    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = eig(bhb_matrix1, BB_matrix)
    
    return eigenvectors, eigenvalues


def cal_V_ee_matrix(c, bhb_matrix, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, Gauss_mat, parameter1, parameter2, l, Z):
    bhb_V_ee_matrix = np.zeros((len(tknot_P) - kord - 2, len(tknot_P) - kord - 2))
    
    for j in range(1, len(tknot_P) - kord - 1):
        for i in range(1, len(tknot_P) - kord - 1):
            if bhb_matrix[j-1, i-1] != 0:
                bhb_V_ee_matrix[j-1, i-1] = cal_bhb_interval1(
                    c, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge,
                    epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, i, j,
                    Gauss_mat, parameter1, parameter2, l, Z
                )
    
    return bhb_V_ee_matrix

# calculate integral of B_iB_j in some interval

def cal_inte_BBinterval(xarray, tknot, kord, index1, index2, Gauss_mat):
    total_sum = 0
    for i in range(len(xarray) - 1):
        a = xarray[i]
        b = xarray[i + 1]
        BB = cal_integral(tknot, kord, index1, index2, fun1, Gauss_mat, a, b, 1)
        total_sum += BB
    return total_sum

# calculate integral of B_iB_j in some interval

def cal_inte_BBinterval1(c, N10, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, Gauss_mat):
    total_sum = 0
    for i in range(len(xarray_P) - 1):
        a = xarray_P[i]
        b = xarray_P[i + 1]
        BB = cal_integral1(c, N10, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, fun1, Gauss_mat, a, b, 1)
        total_sum += BB
        
    return total_sum

# calculate integral of B_iHB_j in some interval

def cal_bhb_interval(xarray, tknot, kord, index1, index2, parameter1, parameter2, Gauss_mat, l, Z):
    total_sum = 0
    for i in range(len(xarray) - 1):
        a = xarray[i]
        b = xarray[i + 1]
        bhb = cal_BHB(tknot, kord, index1, index2, parameter1, parameter2, a, b, Gauss_mat, l, Z)
        total_sum += bhb
        
    return total_sum

# calculate integral of B_iHB_j in some interval

def cal_bhb_interval1(c, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, Gauss_mat, parameter1, parameter2, l, Z):
    total_sum = 0
    for i in range(len(xarray_P) - 1):
        a = xarray_P[i]
        b = xarray_P[i + 1]
        bhb = cal_BHB1(c, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, Gauss_mat, a, b, parameter1, parameter2, l, Z)
        total_sum += bhb
        
    return total_sum

# check integral of ryo

def cal_integral_ryo(ryo, xarray):
    total_sum = 0
    for i in range(len(xarray) - 1):
        a = xarray[i]
        b = xarray[i + 1]
        total_sum += ryo[i] * a**2 * (b - a)

    integral = total_sum * 4 * np.pi
    return integral

# % Bi*Bj*fun
# % Gauss_mat: Gauss with number n
# % a: smallest limit of integral
# % b: largest limit of integral

def cal_integral(tknot, kord, index1, index2, fun, Gauss_mat, a, b, choose):
    total_sum = 0
    N = Gauss_mat.shape[0]  # Get the number of rows in Gauss_mat
    
    for i in range(N):
        transform_coor = Gauss_mat[i, 0] * (b - a) / 2 + (b + a) / 2
        Bavx, dBavx, dB2avx = bsplgen([transform_coor, transform_coor + 1], tknot, kord)  # Assume bsplgen returns the correct values

        if choose == 1:
            total_sum += Bavx[index1] * Bavx[index2] * fun(transform_coor) * Gauss_mat[i, 1]
        else:
            total_sum += dBavx[index1] * dBavx[index2] * fun(transform_coor) * Gauss_mat[i, 1]

    integral = (b - a) / 2 * total_sum
    return integral



def cal_integral1(c, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, fun, Gauss_mat, a, b, choose):
    total_sum = 0
    N = Gauss_mat.shape[0]  # Get the number of rows in Gauss_mat

    for i in range(N):
        transform_coor = Gauss_mat[i, 0] * (b - a) / 2 + (b + a) / 2

        Bavx, dBavx, dB2avx = bsplgen([transform_coor, transform_coor + 1], tknot_P, kord)  # Assuming bsplgen returns the correct values
        
        B_Gaussian = Bavx[0, :]
        result = np.dot(B_Gaussian, c)
        Pot = result / transform_coor
        V_ee_dir = Pot

        P10_Gaussian = Bavx[0, 1:-1] @ c0[:, 0]
        P20_Gaussian = Bavx[0, 1:-1] @ c0[:, 1]
        P21_Gaussian = Bavx[0, 1:-1] @ c1[:, 0]
        P30_Gaussian = Bavx[0, 1:-1] @ c0[:, 2]
        P31_Gaussian = Bavx[0, 1:-1] @ c1[:, 1]
        P32_Gaussian = Bavx[0, 1:-1] @ c2[:, 0]
        P40_Gaussian = Bavx[0, 1:-1] @ c0[:, 3]

        ryo_ne_Gaussian = (1 / (4 * np.pi)) * charge * (
            N10 * (P10_Gaussian / transform_coor) ** 2 +
            N20 * (P20_Gaussian / transform_coor) ** 2 +
            N21 * (P21_Gaussian / transform_coor) ** 2 +
            N30 * (P30_Gaussian / transform_coor) ** 2 +
            N31 * (P31_Gaussian / transform_coor) ** 2 +
            N32 * (P32_Gaussian / transform_coor) ** 2 +
            N40 * (P40_Gaussian / transform_coor) ** 2
        )

        V_ee_exch = -3 * charge / (4 * np.pi * epsilon0) * (3 * ryo_ne_Gaussian / (charge * 8 * np.pi)) ** (1 / 3)

        V_ee = V_ee_dir + V_ee_exch

        if choose == 1:
            total_sum += Bavx[0, index1] * Bavx[0, index2] * fun(transform_coor) * Gauss_mat[i, 1]
        elif choose == 2:
            total_sum += dBavx[0, index1] * dBavx[0, index2] * fun(transform_coor) * Gauss_mat[i, 1]
        else:
            total_sum += Bavx[0, index1] * Bavx[0, index2] * V_ee * Gauss_mat[i, 1]

    V_integral = (b - a) / 2 * total_sum
    return V_integral

def fun1(x):
    return 1

def fun2(x):
    return 1 / x**2

def fun3(x):
    return 1 / x

# bhb include V_ee

def cal_BHB1(c, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, Gauss_mat, a, b, parameter1, parameter2, l, Z):
    item4 = charge * cal_integral1(c, c0, c1, c2, N10, N20, N21, N30, N31, N32, N40, charge, epsilon0, xarray_P, xarray_V, tknot_V, tknot_P, kord, index1, index2, fun1, Gauss_mat, a, b, 3)
    
    bhb = item4
    return bhb


def cal_BHB(tknot, kord, index1, index2, parameter1, parameter2, a, b, Gauss_mat, l, Z):
    item1 = parameter1 * cal_integral(tknot, kord, index1, index2, fun1, Gauss_mat, a, b, 2)
    item2 = parameter1 * l * (l + 1) * cal_integral(tknot, kord, index1, index2, fun2, Gauss_mat, a, b, 1)
    item3 = -Z * parameter2 * cal_integral(tknot, kord, index1, index2, fun3, Gauss_mat, a, b, 1)
    
    bhb = item1 + item2 + item3
    return bhb

# % row of Bavx: the value of Bi at a knotpoint
# % column of Bavx: basis function

def bsplgen(xintervall, tknot, kord):
    istart = 0
    islut = len(tknot) - kord

    punkter = len(xintervall)
    Bavx = np.zeros((punkter, islut))
    dBavx = np.zeros((punkter, islut))
    dB2avx = np.zeros((punkter, islut))
    
    xnr = 0

    for x in xintervall:
        B = np.zeros((len(tknot), kord))
        xnr += 1

        for k in range(1, kord + 1):
            for i in range(istart, islut):
                if k == 1:
                    if tknot[i] < x < tknot[i + 1]:
                        B[i, 0] = 1
                    elif tknot[i] == x < tknot[i + 1]:
                        B[i, 0] = 1
                    else:
                        B[i, 0] = 0
                elif k > 1:
                    if i < kord - k + 1:  
                        B[i, k - 1] = 0
                    elif B[i, k - 2] == 0 and tknot[i + k - 1] - tknot[i] == 0:
                        if B[i + 1, k - 2] == 0:
                            B[i, k - 1] = 0
                        else:
                            B[i, k - 1] = (tknot[i + k] - x) / (tknot[i + k] - tknot[i + 1]) * B[i + 1, k - 2]
                            if k == kord:
                                dBavx[xnr - 1, i] = (k - 1) * (0 - B[i + 1, k - 2] / (tknot[i + k] - tknot[i + 1]))
                    elif B[i + 1, k - 2] == 0 and tknot[i + k] - tknot[i + 1] == 0:
                        B[i, k - 1] = (x - tknot[i]) / (tknot[i + k - 1] - tknot[i]) * B[i, k - 2]
                        if k == kord:
                            dBavx[xnr - 1, i] = (k - 1) * (B[i, k - 2] / (tknot[i + k - 1] - tknot[i]) - 0)
                    else:
                        B[i, k - 1] = (x - tknot[i]) / (tknot[i + k - 1] - tknot[i]) * B[i, k - 2] + \
                                      (tknot[i + k] - x) / (tknot[i + k] - tknot[i + 1]) * B[i + 1, k - 2]
                        if k == kord:
                            dBavx[xnr - 1, i] = (k - 1) * (B[i, k - 2] / (tknot[i + k - 1] - tknot[i]) -
                                                            B[i + 1, k - 2] / (tknot[i + k] - tknot[i + 1]))
    if k == kord:
          if B[i, k - 2] != 0:
             dB2avx[xnr - 1, i] = (k - 1) * (k - 2) * B[i, k - 2] / \
                             ((tknot[i + k - 1] - tknot[i]) * (tknot[i + k - 2] - tknot[i]))

    if B[i + 1, k - 2] != 0:
        dB2avx[xnr - 1, i] -= (k - 1) * (k - 2) * (
            B[i + 1, k - 2] / ((tknot[i + k - 1] - tknot[i]) * (tknot[i + k - 1] - tknot[i + 1])) +
            B[i + 1, k - 2] / ((tknot[i + k] - tknot[i + 1]) * (tknot[i + k - 1] - tknot[i + 1]))
        )

    if B[i + 2, k - 2] != 0:
        dB2avx[xnr - 1, i] += (k - 1) * (k - 2) * B[i + 2, k - 2] / \
                              ((tknot[i + k] - tknot[i + 1]) * (tknot[i + k] - tknot[i + 2]))
        

        for indexi in range(istart, islut):
         if (xnr == len(xintervall) and indexi == islut - 1):
            Bavx[xnr - 1, indexi] = 1
            
            dBavx[xnr - 1, indexi] = (kord - 1) / (tknot[len(tknot) - 1] - tknot[islut])
            
            dB2avx[xnr - 1, indexi] = (kord - 1) * (kord - 2) / \
                ((tknot[len(tknot) - 1] - tknot[islut]) * (tknot[len(tknot) - 2] - tknot[islut]))

    elif (xnr == len(xintervall) and indexi == islut - 2):
        dBavx[xnr - 1, indexi] = -(kord - 1) / (tknot[len(tknot) - 1] - tknot[islut])
        dB2avx[xnr - 1, indexi] = -(kord - 1) * (kord - 2) / \
            ((tknot[len(tknot) - 1] - tknot[islut]) * (tknot[len(tknot) - 2] - tknot[islut])) \
            - (kord - 1) * (kord - 2) / \
            ((tknot[len(tknot) - 1] - tknot[islut - 1]) * (tknot[len(tknot) - 2] - tknot[islut]))

    elif (xnr == len(xintervall) and indexi == islut - 3):
        dB2avx[xnr - 1, indexi] = (kord - 1) * (kord - 2) / \
            ((tknot[len(tknot) - 1] - tknot[islut - 1]) * (tknot[len(tknot) - 1] - tknot[islut]))

    else:
        Bavx[xnr - 1, indexi] = B[indexi, kord - 1]
	    
