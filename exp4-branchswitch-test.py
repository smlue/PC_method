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
    return math.sqrt(sum(x**2 for x in vector))

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
'''
def choose_gradient(f, dfB, p: float, u, h: float, sgn: int):       
    best = (dfB[0], math.inf)
    for grad in dfB:
        tangent = induced_tangent(grad)
        #pred = predictor(u, tangent, h, sgn)
        val = f(pred[0], pred[1]) - p 
        if math.fabs(val) < best[1]:       
            best = (grad, math.fabs(val))
    return best[0]
'''


def corrector(function, dB, xk: float, tk: float, p: float, epsilon: float=10e-10, max_iteration: int=1000):
    m = 0
    ym = xk
    u = tk
    while m < max_iteration:
        if math.fabs(function(ym, u) - p) <= epsilon:
            return [ym, True]
        dBf = dB(ym, u)
        Am = dBf[0][0]
        if Am == 0.0:
            return [0.0, False]
        ym = ym - (function(ym, u) - p)/Am
        m += 1
    return [0.0, False]

'''
function is a lipschitz mapping from R^2 to R
dB is the bouligand differential of function
'''

def alg(function, dB, x0: float, t0: float, tmax: float, steplength: float, eta: float, tau: float, r: float, hmin: float, directions ,max_iter:int=10000, p:float=0.0, epsilon: float =10e-10):

    if tmax <= t0:
        raise ValueError('tmax is smaller then t0')
    
    if math.fabs(function(x0, t0) - p) > epsilon:
        raise ValueError(f'The point ({x0}, {t0}) is not a good approximation!')

    mode = 'FOLLOW'    # mode set to True
    h = steplength
    tk = t0
    xk = x0
    solution = [[x0, t0]]

    while tk < tmax:      # iterates a maxim amount of 'max_iter'

        fail = True
        while fail:                     # u is the next
            delta = min(h, tmax-tk)
            u = tk + delta              
            correction = corrector(
                function, dB, xk, u, p, epsilon, max_iter)
            
            if correction[1]:    
                fail = False
                x_plus = correction[0]
            else:
                fail = True
                u -= delta
                h = delta/2
            if h < hmin:
                raise Exception(
                    '!Method did not converge! Failed by making too small of a steplength!'
                    )
                        
        for d in directions:
            bif_correction = corrector(function, dB, x_plus + r*d, u, p, epsilon, max_iter)
            if not bif_correction[1]:       # Discarding failed corrections
                continue
            omega = bif_correction[0]
            print(f'omega: {omega}')
            if math.fabs(omega - x_plus) > tau:
                x_plus = omega
                mode = 'LEAVE'
                break

        xk = x_plus
        tk = u
        solution.append([xk, tk])
        if mode == 'LEAVE':
            mode = 'FOLLOW'

    return solution

if __name__ == '__main__':

    def f1(x, t):
        return x**3 -t*x
    
    def dBf1(x, t):
        return [[3*x**2-t, -x]]

    #points1 = alg(f1, dBf1, x0=0, t0=-0.025, tmax=0.025, steplength=0.01, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-4, tau=10e-10, r=0.05, directions=[1, -1])  
    #graph(points1)   # MIGHT WANT TO CHECK HOW THIS WORKS --> CHECKED, WORKS

    def sgn(x: float):
        if x == 0.0:
            return 0.0
        elif x > 0:
            return 1
        else:
            return -1

    def f2(x, t):
        return min(abs(x), abs(x-t))

    def dBf2(x, t):
        if abs(x) == abs(x-t):
            if x == 0 and t == 0:
                return [[1, 0],
                        [-1, 0],
                        [1, -1],
                        [-1, -1]]
            elif t > 0:
                return [[1, 0],
                        [-1, -1]]
            else:
                return [[-1, 0],
                        [1, -1]]
        elif abs(x) < abs(x-t):
            if x == 0:
                return [[1, 0],
                        [-1, 0]]
            return [[sgn(x), 0]]
        else:
            if x == t:
                return [[1, -1],
                        [-1, 1]]
            return [[sgn(x-t), -sgn(x-t)]]
        
    def f3(x, t):
            return min(abs(x), abs(x-t-1/math.pi))
    
    def dBf3(x, t):
        if x == 0 and t == -1/math.pi:
            return [[1, 0],
                    [-1,0],
                    [1, -1],
                    [-1, 1]]
        if t < -1/math.pi and x > t/2 + 1/(2*math.pi):
            if x != 0:
                return [[sgn(x), 0]]
            else:
                return [[-1,0], [1, 0]]
        if t < -1/math.pi and x < t/2 + 1/(2*math.pi):
            if x-t-1/math.pi != 0:
                return [[sgn(x-t-1/math.pi), -sgn(x-t-1/math.pi)]]
            else:
                return [[1,-1],[-1, 1]]
        if t > -1/math.pi and x < t/2 + 1/(2*math.pi):
            if x != 0:
                return [[sgn(x), 0]]
            else:
                return [[-1,0], [1, 0]]
        if t > -1/math.pi and x > t/2 + 1/(2*math.pi):
            if x-t-1/math.pi != 0:
                return [[sgn(x-t-1/math.pi), -sgn(x-t-1/math.pi)]]
            else:
                return [[1,-1],[-1, 1]]
        if t == -1/math.pi:
            if x > 0:
                return [[1, 0], [1, -1]]
            if x < 0:
                return [[-1, 0],[-1, 1]]
        if x == t/2 + 1/(2*math.pi):
            if t < -1/math.pi:
                return [[1, 0],[-1, 1]]
            if t > -1/math.pi:
                return [[-1, 0],[1, -1]]

    #points3 = alg(f3, dBf3, x0=0, t0=-1, tmax=0.4, steplength=0.1, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])
    #graph(points3) 

'''
    # TESTING WHEN NEW BRANCH IS FOUND
    # TEST1 - steplength=0.1
    test1 = alg(f2, dBf2, x0=0, t0=-0.35, tmax=0.35, steplength=0.1, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])  
    graph(test1)

    #TEST6 - steplength=0.125
    test6 = alg(f2, dBf2, x0=0, t0=-0.4, tmax=0.4, steplength=0.125, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])  
    graph(test6)

    #TEST5 - steplength=0.15
    test5 = alg(f2, dBf2, x0=0, t0=-0.8, tmax=0.8, steplength=0.15, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])  
    graph(test5)

    # TEST4 - steplength=0.2
    test4 = alg(f2, dBf2, x0=0, t0=-0.9, tmax=0.9, steplength=0.2, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])  
    graph(test4)    

    # TEST2 - steplength=0.5
    test2 = alg(f2, dBf2, x0=0, t0=-1.75, tmax=1.75, steplength=0.5, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])  
    graph(test2)

    # TEST3 - steplength=1
    test3 = alg(f2, dBf2, x0=0, t0=-5.5, tmax=5.5, steplength=1, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-2, tau=10e-3, r=0.1, directions=[1, -1])  
    graph(test3)
'''
    

    

    






    


            

    