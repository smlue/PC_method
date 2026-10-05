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
            correction = corrector(function, dB, xk, u, p, epsilon, max_iter)
            if correction[1]:    
                fail = False
                x_plus = correction[0]
            else:
                fail = True
                u -= delta
                h = delta/2
            if h < hmin:
                raise Exception('!Method did not converge! Failed by making too small of a steplength!')
            
        B = dB(xk, tk)[0][0]
        Bnew = dB(x_plus, u)[0][0]
        if mode == 'FOLLOW' and (B*Bnew < 0 or min(abs(B), abs(Bnew)) <= eta or abs(u) < 0.01):   # Alarm that a bifurcation might have occured
            print('Got to SEARCH for bifurcation before accepting the new point.')
            print(f'FIRST: xk is: {xk}, tk is {tk}')
            print(f'SECOND: x_plus is {x_plus}, u is {u}')
            mode = 'SEARCH'

        if mode == 'SEARCH':
            for d in directions:
                bif_correction = corrector(function, dB, x_plus + r*d, u, p, epsilon, max_iter, )
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
        Ak = dB(xk, tk)[0][0]
        print(f'THIRD: x is: {xk}, t is {tk}')
        print(f'derivative at the point is {Ak}')
        print()

        if mode == 'LEAVE' and Ak > 2*eta:
            mode = 'FOLLOW'

    return solution

if __name__ == '__main__':

    def f1(x, t):
        return x**3 -t*x
    
    def dBf1(x, t):
        return [[3*x**2-t, -x]]

    #points1 = alg(f1, dBf1, x0=0, t0=-0.015, tmax=0.05, steplength=0.01, hmin=10e-6, max_iter=30, epsilon=10e-10, eta=10e-3, tau=10e-3, r=0.1, directions=[-1, 1])
    #print(points1)   
    #graph(points1)   # WORKS!!

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
                        [-1, -1]]
            return [[sgn(x-t), -sgn(x-t)]]
        
    points2 = alg(f2, dBf2, x0=0, t0=-0.3, tmax=0.3, steplength=0.1, hmin=10e-6, max_iter=100, epsilon=10e-10, eta=10e-3, tau=10e-3, r=0.5, directions=[1, -1])  
    graph(points2)

    def f3(x, t):
        return None


            

    