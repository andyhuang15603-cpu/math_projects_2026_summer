import numpy as np 
import matplotlib.pyplot as plt

def Func(t, var):
    return (var[0] - 1 / 3 * (var[0] ** 3) - var[1], var[0])

def RungeKutta(num, x_0 = 1, y_0 = 1):
    len = 100 / num
    t = np.linspace(0, 100, num + 1)
    var = [(x_0, y_0)]
    e = [(0, 0)]
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    for i in range(num):
        k = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype = float)
        for j in range(7):
            k[j] = Func(t[i] + len * C[j], var[i] + len * np.sum(k * np.expand_dims(A[j], axis = 1), axis = 0))
        
        new_var_4 = var[i] + len * np.sum(np.expand_dims(B4, axis = 1) * k, axis = 0)
        new_var_5 = var[i] + len * np.sum(np.expand_dims(B5, axis = 1) * k, axis = 0)
        e.append(np.abs(new_var_4 - new_var_5))
        var.append(new_var_5)

    var = np.array(var)
    e = np.array(e)
    return (len, t, var, e)

def Plot(len, t, var, e):
    fig, ax = plt.subplots(2, 2, figsize = (16, 12))
    
    ax[0, 0].plot(t, var[:, 0], color = 'red', label = 'X', ls = '-')
    ax[0, 0].plot(t, var[:, 1], color = 'blue', label = 'Y', ls = '-')
    ax[0, 0].set_title(f"X and Y versus time with step of {len}") 
    ax[0, 0].set_ylabel("X and Y")
    ax[0, 0].set_xlabel("time")
    ax[0, 0].grid(True)
    ax[0, 0].legend()

    ax[0, 1].plot(t, e[:, 0], color = 'green', label = 'X', ls = '-')
    ax[0, 1].plot(t, e[:, 1], color = 'purple', label = 'Y', ls = '-')
    ax[0, 1].set_title(f"Error of X and Y versus time with step of {len}") 
    ax[0, 1].set_ylabel("Error of X and Y")
    ax[0, 1].set_xlabel("time")
    ax[0, 1].grid(True)
    ax[0, 1].legend()

    ax[1, 0].plot(var[:, 0], var[:, 1], color = 'red', label = 'Phase')
    ax[1, 0].set_title(f"Phase Diagram of X and Y with step of {len}") 
    ax[1, 0].set_ylabel("X")
    ax[1, 0].set_xlabel("Y")
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    ax[1, 1].axis('off')

    fig.tight_layout()
    plt.show()

def main():
    len, t, var, e = RungeKutta(1000)
    Plot(len, t, var, e)
    len, t, var, e = RungeKutta(10000)
    Plot(len, t, var, e)
    len, t, var, e = RungeKutta(100000)
    Plot(len, t, var, e)
    

if __name__ == "__main__":
    main()