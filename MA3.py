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

    # skapar n slumpmässiga punkter i kvadraten -1,1 * -1,1
    for _ in range(n): 
        x = random.uniform(-r, r)
        y = random.uniform(-r, r)
        dist = (x**2 + y**2)**0.5
        if dist <= 1: # se om den är inuti cirklen 
            inside_x.append(x)
            inside_y.append(y)
        else: 
            outside_x.append(x)
            outside_y.append(y)

    # uppskatta pi mha av formel
    nc = len(inside_x)
    pi = 4*nc/n # 4 = kvadratens area
    print(f'pi = {pi}')

    # skapa en plot
    plt.figure()
    plt.scatter(inside_x, inside_y, c ='r', s = 1, label = 'inside')
    plt.scatter(outside_x, outside_y, c = 'b', s = 1, label = 'outside')
    plt.title('Approximation of pi using Monte Carlo')
    plt.legend()
    plt.savefig(f'cirkel_{n}.png')
    plt.show()
    return pi

# Exc2, approximation
def sphere_volume(n, d, r=1): 
    # n is the number of points
    # d is the number of dimensions of the sphere
    inside_count = 0

    for i in range(n):
        dot = [random.uniform(-r, r) for _ in range(d)] # skapar en slumpmässig punkt med d dimentsioner
        squares = list(map(lambda x: x**2, dot)) # gör en lista med x^2 där x är varje dot map gör att x blit varje tal fårn dot
        tot_sum = functools.reduce(lambda a, b: a+b, squares) # summerar kvadraterna

        if tot_sum <= r**2: # när summan för punkten i den dim har skapats kan vi se om den ör innuti svären
            inside_count += 1

    approx_volume = (2*r)**d*(inside_count/n) # uppskatta volymen mha formel

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
    # måste optimera koden för numba 
    inside_count = 0

    for _ in range(n):
        total_sum = 0.0

        # generar en punk med d kordinater
        for _ in range(d):
            x = np.random.uniform(-r, r)
            total_sum += x**2 # beräknar summan av kordinat**2

        if total_sum <= r**2: 
            inside_count += 1

    approx_volume = (2*r)**d*(inside_count/n) 
    
    return approx_volume

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10, r=1):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes

    n_per_np = n//np # hur många gångr ska varje processor göra
    remainder = n % np # om de ej är jämt delbart

    tasks = [(n_per_np + (1 if i < remainder else 0), d, r) for i in range(np)]
    
    # kör berkningarna parallellt
    with future.ProcessPoolExecutor() as ex:
        results = ex.map(worker_func, tasks)
        tot_inside_count = sum(results)

    approx_volume = (2*r)**d * (tot_inside_count/n)
    return approx_volume

def worker_func(arg):
    n, d, r = arg
    inside_count = 0
    for _ in range(n):
            total_sum = 0.0
    
            for _ in range(d):
                x = random.uniform(-r, r)
                total_sum += x**2
    
            if total_sum <= r**2: 
                inside_count += 1 

    return inside_count # retunerar antalet punkter
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    print(f'Estimatet volume of {d} dimentional sphere = {sphere_volume(n, d):.4}')
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d):.4}")

    n = 100000
    d = 11

    print(f'Estimatet volume of {d} dimentional sphere = {sphere_volume(n, d):.4}')
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d):.4}")

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

    print(f"Exc4: Sequential time of {d} and {n}: {stop-start:.4}")

    # Parallell version
    start = pc()
    par_vol = sphere_volume_parallel(n, d)
    stop = pc()
    par_time = stop - start
    print(f"Parallel time with processes: {par_time:.4f} seconds")

    
    

if __name__ == '__main__':
	main()
