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

def sign(x: float):
    if x == 0.0:
        return 0.0
    elif x > 0:
        return 1
    else:
        return -1
    
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
    return (A[0]/AAT, A[1]/AAT)     # mathematically a column vector is returned

def predictor(u, tangent, h: float, sgn: int):
    return (u[0] + sgn*h*tangent[0], u[1] + sgn*h*tangent[1])

def corrector():
    pass

# Chooses the gradient that will continue traversing the path with the same sign
def choose_gradient(f, dfB, p: float, u, h: float, sgn: int):       
    best = (dfB[0], math.inf)
    for grad in dfB:
        tangent = induced_tangent(grad)
        pred = predictor(u, tangent, h, sgn)
        val = f(pred[0], pred[1]) - p 
        if math.fabs(val) < best[1]:
            best = (grad, val)
    return best[0]

#Function takes as INPUT a point and outputs the VALUE
#Derivative takes as INPUT a point and outputs a list of length 2 as that gradient of the function

def bifurcation_alg(function, derivative, x_0: float, t_0: float, interval, sign: int, tolerance: float):        # (t_0, x_0) must be a regular point of func = 0

    if abs(function(x_0, t_0)) > tolerance :      
        raise ValueError(f'The point ({x_0}, {t_0}) is not a good approximation for a zero point of the function.')
    
    if norm(derivative(x_0, t_0)[0]) < tolerance: #Taking the first gradient for example --> shoudl change to loop
        raise ValueError(f'The point ({x_0}, {t_0}) is not a regular point of the function.')
    
    solution = []

    u = (x_0, t_0)
    sgn = sign

    solution.append(u)
    
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
        
            dB_u = derivative(u[0], u[1])     #Returns Bouligand subdifferential as List of tuples representing the gradients
            der_u = dB_u[0]
            if len(dB_u) != 1:
                print('It got here 1')
                der_u = choose_gradient(function, dB_u, p, u, h, sgn)
            tan_u = induced_tangent(der_u)        
            v = predictor(u, tan_u, h, sgn)    # predictor step

            convergence = False
            m = 0
            while not convergence and m <= max_corrector:         # will get next approximation point of the curve
                v_x = v[0]
                v_t = v[1]
                func_v = function(v_x, v_t) 
                dB_v = derivative(v_x, v_t)
                der_v = dB_v[0]
                if len(dB_v) != 1:
                    der_v = choose_gradient(function, dB_v, p, v, h, sgn)
                mp = moore_penrose(der_v)
                v = (v_x - mp[0]*(func_v - p), v_t - mp[1]*(func_v - p))    # corrector steps
                if abs(function(v[0], v[1]) - p) <= tolerance:
                    convergence = True
                    continue
                m += 1

            if v[1] > interval[1] or v[1] < interval[0]:
                return solution
            
            #BIFURCATION CHECK:
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


if __name__ == '__main__':

    #GRAPH OF CIRCLE:
    '''
        def circle(x: float, t: float): 
            return x**2 + t**2 - 1

        def deriv_circle(x: float, t: float):
            return (2*x, 2*t)
        
        circle1 = bifurcation_alg(circle, deriv_circle, 0, 1, [-1 , 1], sign=-1, tolerance=10e-5)  
        graph(circle1)
    '''
    #GRAPH OF SIMPLE BIFURCATION POINT
    '''
    def bifurcation_point(x: float, t: float) -> float:
        return x**3 - t*x

    def der_bifurcation(x: float, t: float):
        return [(3*x**2 - t, -x)]
    '''
    #GRAPH OF SIMPLE EXAMPLE:
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

    example1 = bifurcation_alg(example, der_example, 0, 0, [-1, 1], sign=1, tolerance=10e-5)
    graph(example1)

    pass
