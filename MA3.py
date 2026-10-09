""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import numpy as np
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import functools
from numba import njit

# Exc1
def approximate_pi(n):
    # n is the number of points
    print(f' for n = {n}')
    r = 1 # sätt radien till 1
    # Write your code here
    inside_x, inside_y = [], []
    outside_x, outside_y = [], []

    for _ in range(n): 
        x = random.uniform(-r, r)
        y = random.uniform(-r, r)
        dist = (x**2 + y**2)**0.5
        if dist <= 1: 
            inside_x.append(x)
            inside_y.append(y)
        else: 
            outside_x.append(x)
            outside_y.append(y)

    # beräkna pi
    nc = len(inside_x)
    pi = 4*nc/n
    print(f'pi = {pi}')

    # skapa en plot
    plt.figure()
    plt.scatter(inside_x, inside_y, c ='r', s = 1, label = 'inside')
    plt.scatter(outside_x, outside_y, c = 'b', s = 1, label = 'outside')
    plt.title('Approximation of pi using Monte Carlo')
    plt.legend()
    plt.show()
    return pi

# Exc2, approximation
def sphere_volume(n, d, r=1): 
    # n is the number of points
    # d is the number of dimensions of the sphere
    inside_count = 0

    for i in range(n):
        dot = [random.uniform(-r, r) for _ in range(d)] # skapar en punkt med d dimentsioner
        squares = list(map(lambda x: x**2, dot)) # gör en lista med x^2 där x är varje dot
        tot_sum = functools.reduce(lambda a, b: a+b, squares) # summerar kvadraterna

        if tot_sum <= 1: 
            inside_count += 1

    approx_volume = 2**d*(inside_count/n)

    return approx_volume

#Exc2, real value
def hypersphere_exact(n, d, r=1):
    # n is the number of points
    # d is the number of dimensions of the sphere 

    exact_volume = np.pi**(d/2)/m.gamma((d/2)+1)*r**d


    return exact_volume

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int, r = 1)->float:
    # n is the number of points
    #  # d is the number of dimensions of the sphere
    inside_count = 0

    for _ in range(n):
        total_sum = 0.0

        for _ in range(d):
            x = np.random.uniform(-1, 1)
            total_sum += x**2

        if total_sum <= 1: 
            inside_count += 1

    approx_volume = 2**d*(inside_count/n)
    
    return approx_volume

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, r =1, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes

    n_per_np = n//np # hur många gångr ska varje processor göra
    tasks = [(n_per_np, d, r) for _ in range(np)]

    with future.ProcessPoolExecutor() as ex:
        results = ex.map(worker_func, tasks)
        tot_inside_count = sum(results)

    approx_volume = 2**d * (tot_inside_count/n)
    return approx_volume

def worker_func(arg):
    n, d, r = arg
    inside_count = 0
    for _ in range(n):
            total_sum = 0.0
    
            for _ in range(d):
                x = random.uniform(-1, 1)
                total_sum += x**2
    
            if total_sum <= 1: 
                inside_count += 1
    
        

    return inside_count
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    print(f'Estimatet volume of {d} dimentional sphere = {sphere_volume(n, d)}')
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11

    print(f'Estimatet volume of {d} dimentional sphere = {sphere_volume(n, d)}')
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11

    print(f"Exc3: Sequential time of {d} and {n}: ")
    
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"First version {i+1}: {stop - start:.4f} seconds")
    
    
    for i in range(3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Numba time {i+1}: {stop - start:.4f} seconds")
    

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")

    print("What is parallel time?")

    
    

if __name__ == '__main__':
	main()
