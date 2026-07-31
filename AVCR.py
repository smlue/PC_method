import sympy as sy
import numpy as np
import matplotlib.pyplot as plt 
import math

#assuming jacobian(gradient) has full rank 1 --- NOT TRUE!! a bifurcation point will not have full rank --> check for in code

x, t = sy.symbols("x t")

h = 1
epsilon = 0.01      # epsilon must be 'small enough' in order for the method to work correctly
p = 10e-2     # perturbation vector (in this case a number)
 
def function(func: str, x_0: float, t_0: float):
    f = sy.sympify(func)
    value = sy.lambdify((x, t), f, 'numpy')
    return value(x_0, t_0)

def derivative(func: str, x_0: float, t_0: float):
    f = sy.sympify(func)
    d_x = sy.diff(f, x)
    d_t = sy.diff(f, t)
    val_x = sy.lambdify((x, t), d_x, 'numpy')
    val_t = sy.lambdify((x, t), d_t, 'numpy')
    return  (val_x(x_0, t_0), val_t(x_0, t_0))

def norm(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

def induced_tangent(der):
    tan = (der[0], -der[1])
    n = norm(tan)

    if n != 0:
        tan = (tan[0]/n, tan[1]/n)

    det = der[0]*tan[1] - der[1]*tan[0]
    if det > 0:
        return tan
    
    return (-tan[0], -tan[1])

def moore_penrose(A):         # finds the moore-penrose inverse of 1x2 matrix A - A is a list 
    if norm(A) == 0:
        raise ValueError(str(A) + ' does not have maximal rank one.')
    AAT = A[0]**2 + A[1]**2
    return (A[0]/AAT, A[1]/AAT)     # mathematically a column vector is returned

    
def bifurcation_alg(func: str, x_0: float, t_0: float, interval, sign: int, deriv: str = None):        # (t_0, x_0) must be a regular point of func = 0

    if abs(function(func, x_0, t_0)) > epsilon :
        raise ValueError('The point ' + '(' + str(x_0) + ', ' + str(t_0) + ') ' + 'is not a good approximation of a zero point of ' + func)

    solution = []
    
    func_0 = function(func, x_0, t_0)
    der_0 = derivative(func, x_0, t_0)
    u = (x_0, t_0)
    tangent = induced_tangent(der_0)
    solution.append(u)
    sgn = sign

    if der_0[0] == 0 and der_0[1] == 0:
            raise ValueError('The point ' + '(' + str(x_0) + ', ' + str(t_0) + ') ' + 'is not a regualr point of ' + func)
        
    cd = False          # logical variable for reversing direction
    traversing = True

    while traversing:

        u_x = u[0]
        u_t = u[1]
        der_u = derivative(func, u_x, u_t)
        v = (u_x + h*sgn*tangent[0], u_t + h*sgn*tangent[1])    # predictor step
        convergence = False


        while not convergence:
            v_x = v[0]
            v_t = v[1]
            func_v = function(func, v_x, v_t) 
            der_v = derivative(func, v_x, v_t)
            mp = moore_penrose(der_v)
            v = (v_x - mp[0]*(func_v - p), v_t - mp[1]*(func_v - p))    # corrector step
            if abs(function(func, v[0], v[1])) <= epsilon:
                 convergence = True

        tan_u = induced_tangent(der_u)
        tan_v = induced_tangent(der_v)
        if tan_u[0]*tan_v[0] + tan_u[1]*tan_v[1] < 0:
            sgn = -sgn
            print("Simple bifurcation points detected between " + str(u) + " and " + str(v))
            traversing = bool(input('Would you like to end the program? True - keep going, False - end: '))
            if traversing == False:
                solution.append(v)
                return solution
            cd = bool(input('Would you like to change directions in traversing? True/False: '))
            if cd == True:
                sgn = -1*sgn
        
        u = v
        print(u)
        # Here adapt steplength algorithm

        if u[1] > interval[1] or u[1] < interval[0]:
            return solution
        
        solution.append(u)

    return solution

def graph_implicit(points):       # points - Nx2 list of floats
    t = []
    x = []
    for point in points:
        t.append(point[1])
        x.append(point[0])
    t = np.array(t)
    x = np.array(x)
    fig, axs = plt.subplots(figsize=(5, 5), layout='constrained') 
    axs.plot(t, x, 'o', markersize=3)  
    axs.set_xlabel('t - parameter')
    axs.set_ylabel('x')
    plt.show()


if __name__ == '__main__':

    #Implicitly defined functions without bifurcation points:
    #points1 = bifurcation_alg('x**2 + t**2 - 25', 4.89898, 1,[0, 4.8], -1)     # this works!!!
    #graph_implicit(points1)

    #points2 = bifurcation_alg('x*sin(x) - t', 0.43, 0.17925, 1, [-1, 1.7], -1)
    #graph_implicit(points2)
    
    #points4_ = bifurcation_alg('x - t', -0.5, -0.5, [-1, 1], sign=1)
    #graph_implicit(points4_)

    # SPECIAL POINTS - crossing over maxima/minima, bifurcation points
    #points3 = bifurcation_alg('x**2 + t**2 - 25', 4.89898, -1, [-4.8, 1.2], 1)     
    #graph_implicit(points3)

    # points4 = bifurcation_alg('x**2 - t*x', -0.5, -0.5, [-1, 1], sign=1)
    # print(points4)
    # graph_implicit(points4)

    points5 = bifurcation_alg('x**3 - t*x', 1, 1, [-0.5, 1], sign=-1)
    graph_implicit(points5)


