import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

G = 6.67*10**(-8) #cm^3/g/s^2
c = 3.0*10**10 #cm/s
msun = 1.99 * 10**33 #g
rsun = 6.95 * 10**10 #cm
sec_to_yr = 3.17098*10**(-8)
hubble_time =  14.4 * 10**9 #yr
AU_to_cm = 1.496*10**13 #cm

def lifetime_beta(m1,m2):
    m1 = m1*msun
    m2 = m2*msun
    beta=(64./5.)*((G**3)*m1*m2*(m1+m2))/c**5
    return beta

def lifetime_funcof_a0_in_yrs(m1,m2,a0): #input: a0 in cm
    tc = a0**4./(4.*lifetime_beta(m1,m2))
    return tc*sec_to_yr                  #output: lifetime in years


def gen_samples_using_invcdf(func, minimum, maximum, nsamples):
    """
    Implements inverse cdf sampling for a power function.

    func: pdf to draw samples from

    minimum: minimum value

    maximum: maximum value    
    """
    xarr = np.linspace(minimum, maximum, 50000)
    yarr = func(xarr)
    cumulative_sum = np.cumsum(yarr)
    cumulative_sum *= 1 / cumulative_sum[-1]
    
    interped_inverse_cdf = interp1d(cumulative_sum, xarr)    
    uniform_samples = np.random.uniform(min(cumulative_sum), 1, nsamples)
    
    return interped_inverse_cdf(uniform_samples)