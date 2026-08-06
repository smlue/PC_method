import sympy as sy
import numpy as np
import matplotlib.pyplot as plt 
import math

x, t = sy.symbols("x t")

epsilon = 0.0001      # epsilon must be 'small enough' in order for the method to work correctly
 
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
    tan = (der[1], -der[0])     #OPRAVENO
    n = norm(tan)

    if n != 0:
        tan = (tan[0]/n, tan[1]/n)
    else:
        raise ValueError(str(der) + ' induces zero as its tangent!!')

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

    if abs(function(func, x_0, t_0)) > epsilon :        #TADY TAKY pridat -p ???
        raise ValueError('The point ' + '(' + str(x_0) + ', ' + str(t_0) + ') ' + 'is not a good approximation of a zero point of ' + func)

    solution = []

    u = (x_0, t_0)
    func_0 = function(func, x_0, t_0)
    der_0 = derivative(func, x_0, t_0)
    sgn = sign
    tangent_0 = induced_tangent(der_0)

    solution.append(u)
    

    if der_0[0] == 0 and der_0[1] == 0:
            raise ValueError('The point ' + '(' + str(x_0) + ', ' + str(t_0) + ') ' + 'is not a regualr point of ' + func)

    traversing = True

    while traversing:

        traversing = not bool(input('If you wish to CONTINUE press ENTER otherwise write QUIT: '))
        if traversing == False:
            continue
        h = float(input('Steplength: '))
        max_iterations = int(input('Maximum amount of iterations for predictor: '))
        max_corrector = int(input('Maximum amount of iterations for corrector: '))
        p = float(input('Perturbation vector: '))
        cd = bool(input('If you wish to CHANGE DIRECTION write YES otherwise press ENTER: '))
        print()

        if cd == True:
            sgn *= -1

        n = 0
        while n <= max_iterations:
        
            u_x = u[0]
            u_t = u[1]
            der_u = derivative(func, u_x, u_t)
            tan_u = induced_tangent(der_u)        # OPRAVENO - vypocita se po kazde iteraci derivace znovu
            v = (u_x + h*sgn*tan_u[0], u_t + h*sgn*tan_u[1])    # predictor step
            convergence = False
            
            m = 0
            while not convergence and m <= max_corrector:         # will get next approximation point of the curve
                v_x = v[0]
                v_t = v[1]
                func_v = function(func, v_x, v_t) 
                der_v = derivative(func, v_x, v_t)
                mp = moore_penrose(der_v)
                v = (v_x - mp[0]*(func_v - p), v_t - mp[1]*(func_v - p))    # corrector steps
                if abs(function(func, v[0], v[1]) - p) <= epsilon:
                    convergence = True
                    continue
                m += 1

            if v[1] > interval[1] or v[1] < interval[0]:
                return solution

            tan_v = induced_tangent(der_v)
            if tan_u[0]*tan_v[0] + tan_u[1]*tan_v[1] < 0:
                sgn *= -1
                print("Simple bifurcation point detected between " + str(u) + " and " + str(v))
                u = v           
                solution.append(u)
                n = max_iterations + 1
                continue

            u = v
            solution.append(u)
            n += 1

            # Here adapt step-length algorithm

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
    axs.plot(t, x, 'o', ms=3)  
    axs.set_xlabel('t - parameter')
    axs.set_ylabel('x')
    axs.grid(color = 'grey')
    plt.show()


if __name__ == '__main__':

    #Implicitly defined functions without bifurcation points:
    #points1 = bifurcation_alg('x**2 + t**2 - 25', 3, 4,[-5.5, 4.8], -1)     # THIS WORKS!!
    #graph_implicit(points1)

    #points2 = bifurcation_alg('x*sin(x) - t', 0.43, 0.17925, [-1, 1.7], -1)
    #graph_implicit(points2)
    
    #points3 = bifurcation_alg('x - t', -0.5, -0.5, [-1, 1], sign=1)
    #graph_implicit(points4_)

    # Bifurcation points
    
    #points4 = bifurcation_alg('x**2 - t*x', -0.5, -0.5, [-1, 1], sign=-1)
    #graph_implicit(points4)

    #points5 = bifurcation_alg('x**3 - t*x', 1, 1, [-4 , 1], sign=-1)  #WORKS even for switching branches -- once you switch the direction once you do not       
    #graph_implicit(points5)
    
    pass
