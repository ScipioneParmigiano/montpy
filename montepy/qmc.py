import numpy as np
from scipy.stats import qmc
import math

class QuasiMonteCarloSolver:
    def __init__(self, func, domain):
        """
        Initialize the Quasi-Monte Carlo solver with a function and domain.
        
        Parameters:
            func (callable): The function to integrate.
            domain (tuple or list of tuples): The integration domain. 
                For single-variable functions, domain should be a tuple (a, b).
                For multi-variable functions, domain should be a list of tuples [(a1, b1), (a2, b2), ...].
        """
        self.func = func
        self.domain = np.array(domain)
    
    def integrate_sobol(self, num_samples=1024):
        """
        Perform quasi-Monte Carlo integration using Sobol sequence.
        
        Parameters:
            num_samples (int): The number of quasi-random samples to generate.
        
        Returns:
            float: The estimated integral value.
        """
        samples = self._generate_sobol_samples(num_samples)
        samples = self._transform_samples(samples)
        
        if len(self.domain) == 1:  # Univariate functions
            integral = np.mean(self.func(samples[:, 0])) * (self.domain[0][1] - self.domain[0][0])
        else:  # Multivariate functions
            integral = np.mean(self.func(*samples.T)) * np.prod(self.domain[:, 1] - self.domain[:, 0])
        
        return integral
    
    def integrate_halton(self, num_samples=1000):
        """
        Perform quasi-Monte Carlo integration using Halton sequence.
        
        Parameters:
            num_samples (int): The number of quasi-random samples to generate.
        
        Returns:
            float: The estimated integral value.
        """
        samples = self._generate_halton_samples(num_samples)
        samples = self._transform_samples(samples)
        
        if len(self.domain) == 1:  # Univariate functions
            integral = np.mean(self.func(samples[:, 0])) * (self.domain[0][1] - self.domain[0][0])
        else:  # Multivariate functions
            integral = np.mean(self.func(*samples.T)) * np.prod(self.domain[:, 1] - self.domain[:, 0])
        
        return integral

    
    def integrate_hammersley(self, num_samples=1000):
        """
        Perform quasi-Monte Carlo integration using Hammersley sequence.
        
        Parameters:
            num_samples (int): The number of quasi-random samples to generate.
        
        Returns:
            float: The estimated integral value.
        """
        samples = self._generate_hammersley_samples(num_samples)
        samples = self._transform_samples(samples)
        
        if len(self.domain) == 1:  # Univariate functions
            integral = np.mean(self.func(samples[:, 0])) * (self.domain[0][1] - self.domain[0][0])
        else:  # Multivariate functions
            integral = np.mean(self.func(*samples.T)) * np.prod(self.domain[:, 1] - self.domain[:, 0])
        
        return integral

    
    def _generate_sobol_samples(self, num_samples):
        """
        Generate Sobol sequence samples.
        """
        num_samples = 2 ** math.ceil(math.log2(num_samples))
        # print(num_samples)
        dim = self.domain.shape[1]
        sobol_generator = qmc.Sobol(d=dim, scramble=True)
        samples = sobol_generator.random(num_samples)
        samples = self._transform_samples(samples)
        return samples

    def _generate_halton_samples(self, num_samples):
        """
        Generate Halton sequence samples.
        """
        dim = self.domain.shape[1]
        samples = np.zeros((num_samples, dim))
        for i in range(num_samples):
            samples[i] = self._halton_sample(i + 1, dim)
        return samples
    
    def _generate_hammersley_samples(self, num_samples):
        """
        Generate Hammersley sequence samples.
        """
        samples = np.zeros((num_samples, self.domain.shape[1]))
        for i in range(self.domain.shape[1]):
            samples[:, i] = np.linspace(0, 1, num_samples)
        return samples
    
    def _sobol_sample(self, index, dim):
        """
        Generate a single Sobol sequence sample.
        """
        v = np.zeros(dim)
        c = 1 / (1 << 32)
        for i in range(dim):
            value = 0
            while index:
                if index & 1:
                    break
                value += 1
                index >>= 1
            v[i] = value * c
            index ^= 1
        return v
    
    def _halton_sample(self, index, dim):
        """
        Generate a single Halton sequence sample.
        """
        v = np.zeros(dim)
        for i in range(dim):
            f = 1
            r = 0
            idx = index
            while idx > 0:
                f /= self._primes[i]
                r += f * (idx % self._primes[i])
                idx //= self._primes[i]
            v[i] = r
        return v
    
    def _transform_samples(self, samples):
        """
        Transform samples to fit within the given domain.
        """
        return samples * (self.domain[:, 1] - self.domain[:, 0]) + self.domain[:, 0]

    _primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

