import numpy as np
import pandas as pd
import sys

def rk4(f, y0, t):
    """
    Fourth-order Runge-Kutta method for solving ODEs.
    
    Parameters:
    f : function
        Function that returns the derivative of y at time t.
    y0 : array-like
        Initial condition.
    t : array-like
        Array of time points where the solution is computed."
    """
    y0 = np.asarray(y0)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h/2, y[i-1] + h*k1/2)
        k3 = f(t[i-1] + h/2, y[i-1] + h*k2/2)
        k4 = f(t[i-1] + h, y[i-1] + h*k3)
        y[i] = y[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return y

def newton_raphson(func, dfunc, x0, tol=1e-10, max_iter=1000):
    #print("Starting Newton-Raphson with initial guess:", x0)
    x = x0
    xHist = np.array([x0])
    fxHist = np.array([0])
    dfxHist = np.array([0])
    for i in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_new = x - fx/dfx
        xHist = np.append(xHist, x_new)
        fxHist = np.append(fxHist, fx)
        dfxHist = np.append(dfxHist, dfx)
        #if i % 10 == 0:
        #    print(f"Iteration {i}: x = {x_new}, f(x) = {fx}, f'(x) = {dfx}, step = {x_new - x}")
        if abs(x_new - x) < tol:
            return x_new
        
        if np.isnan(x_new) or np.isinf(x_new):
            
            print("NaN or Inf detected. No solution found. Returning last estimate.")
            df = pd.DataFrame({"x": xHist, "f(x)": fxHist, "f'(x)": dfxHist})
            df.to_csv("newton_raphson_debug.csv", index=False)
            raise ValueError("NaN or Inf encountered in computation.")
            #return x
        if x_new <= 0:
            print("Non-physical negative or zero value encountered. No solution found. Returning last estimate.")
            
            return x
        else:
            x = x_new

    print("Maximum iterations reached. No solution found. Returning last estimate.")
    return x