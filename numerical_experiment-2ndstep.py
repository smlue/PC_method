import numpy as np
import math
import matplotlib.pyplot as plt 

def graph(points):       #Points - Nx2 list of floats
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

def norm(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

def induced_tangent(der):
    tan = (der[1], -der[0])     
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
    return (A[0]/AAT, A[1]/AAT)     

def predictor(u, tangent, h: float, sgn: int):
    return (u[0] + sgn*h*tangent[0], u[1] + sgn*h*tangent[1])

def corrector():
    pass

# Chooses the gradient with the smallest residue
def choose_gradient(f, dfB, p: float, u, h: float, sgn: int):       
    best = (dfB[0], math.inf)
    for grad in dfB:
        tangent = induced_tangent(grad)
        pred = predictor(u, tangent, h, sgn)
        val = f(pred[0], pred[1]) - p 
        if math.fabs(val) < best[1]:       
            best = (grad, math.fabs(val))
    return best[0]

#Function takes as INPUT a point and outputs the VALUE
#Derivative takes as INPUT a point and outputs a LIST OF LENGTH 2 as that gradient of the function

def bifurcation_alg(function, derivative, x_0: float, t_0: float, interval, sign: int, steplength: float, max_predictor: int, max_corrector: int, tolerance: float, switch_branch: bool =False,pert: float=0.0): 

    
    if abs(function(x_0, t_0)) > tolerance :      
        raise ValueError(f'The point ({x_0}, {t_0}) is not a good approximation for a zero point of the function.')
    
    dB_u0 = derivative(x_0, t_0)
    regular = False
    for grad in dB_u0:
        if norm(grad) >= tolerance:
            regular = True
    
    if not regular:        
        raise ValueError(f'The point ({x_0}, {t_0}) is not a regular point of the function.')
        
    solution = []

    u = (x_0, t_0)
    sgn = sign
    h = steplength
    bif_encountered = False
    solution.append(u)
    
    traversing = True
    while traversing:

        p = 0
        max_pred = max_predictor

        if bif_encountered and switch_branch:
            printt = True
            p = pert
            max_pred = 50     # amount of points generated on pertrubation
            sgn *= -1
            bif_encountered = False
            
        n = 0
        while n < max_pred:
            dB_u = derivative(u[0], u[1])     #Returns Bouligand subdifferential as List of tuples representing the gradients
            der_u = choose_gradient(function, dB_u, p, u, h, sgn)
            tan_u = induced_tangent(der_u)        
            v = predictor(u, tan_u, h, sgn)    # predictor step
            
            convergence = False
            m = 0
            while not convergence and m < max_corrector:     
                v_x = v[0]
                v_t = v[1]
                func_v = function(v_x, v_t) 
                dB_v = derivative(v_x, v_t)
                der_v = choose_gradient(function, dB_v, p, v, h, sgn)
                mp = moore_penrose(der_v)
                v = (v_x - mp[0]*(func_v - p), v_t - mp[1]*(func_v - p))    # corrector steps
                if abs(function(v[0], v[1]) - p) <= tolerance:
                    convergence = True
                    continue
                m += 1
                if m == max_corrector:
                    raise RuntimeError(f'The corrector method has not converged in {max_corrector} loop iterations for the tolerance of {tolerance}!')

            if v[1] > interval[1] or v[1] < interval[0]:
                return solution
            
            #BIFURCATION CHECK:
            tan_v = induced_tangent(der_v)
            if tan_u[0]*tan_v[0] + tan_u[1]*tan_v[1] < 0:
                sgn *= -1
                bif_encountered = True
                print("Simple bifurcation point detected between " + str(u) + " and " + str(v))
                u = v
                solution.append(u)
                n = max_pred
                continue

            u = v
            solution.append(u)
            n += 1

            # Here adapt step-length algorithm

    return solution


if __name__ == '__main__':

    def f(x: float, t: float) -> float:
        return x**3 - t*x

    def df(x: float, t: float):
        return [(3*x**2 - t, -x)]

    #first_branch = bifurcation_alg(f, df, 0, 1, [-1, 1], sign=1, steplength=0.01, max_predictor=10000, max_corrector=10000, tolerance=10e-5, swithc_branch=False, pert=0,)  # First branch - line WORKS
    #graph(first_branch)

    second_branch = bifurcation_alg(f, df, 0, 3, [-2, 3], sign=1, steplength=0.1, max_predictor=10000, max_corrector=10000, tolerance=10e-10, switch_branch=True, pert=-10e-20)
    #print(second_branch)
    graph(second_branch)


    #GRAPH OF SIMPLE EXAMPLE:
    '''
    def sign(x: float):
        if x == 0.0:
            return 0.0
        elif x > 0:
            return 1
        else:
            return -1

    def example(x: float, t: float) -> float:
        return math.fabs(x) - t

    def der_example(x: float, t: float):
        if x != 0:
            return [(sign(x),-1)]
        else:
            return [(-1, -1), (1, -1)]

    example1 = bifurcation_alg(example, der_example, 1, 1, [-1, 1], sign=1, steplength=0.1, max_predictor=10000, max_corrector=10000, tolerance=10e-10)
    graph(example1)
    '''
    pass
