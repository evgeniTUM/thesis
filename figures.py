import pylab as pb
pb.ion()
import numpy as np
import GPy


#def gp_regression():
    
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
kernel = GPy.kern.rbf(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X,Y,kernel)

m.plot()
    
m.optimize()


m.plot(plot_raw=True)

def gaussian():
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    import numpy as np

    mean = 0
    variance = 1
    sigma = np.sqrt(variance)

    x = np.linspace(-5, 5, 100)
    plt.figure(figsize=(10, 5))
    plt.plot(x, mlab.normpdf(x, mean, sigma))
    plt.xlim(-5,5)
    plt.ylim(-0.1,0.5)

def test():
    x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    y = 2*x**3 + 5*x**2 + 0.5*x+5
    z = np.polyfit(x, y, 3)

    p = np.poly1d(z) 


    p30 = np.poly1d(np.polyfit(x, y, 30))


    import matplotlib.pyplot as plt
    xp = np.linspace(-2, 6, 100)
    plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')

    plt.ylim(-2,2)

    plt.show()

