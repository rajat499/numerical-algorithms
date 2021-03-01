import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

##Part A
def computeSequence(A, b, X, n):
    m = len(X)
    A = np.flip(A)
    if n<=0:
        print("Invalid Input: "+str(n))
        sys.exit()
    for i in range(m, n):
        X = np.append(X, sum(A*X[-m:])+b)
    return X[:n]

def exactSolution(n):
    return [4**(-i)/3 for i in range(n)]

##Part B
n = 100
fig, ax = plt.subplots(1, 1, figsize =(16, 7))
xticks = np.arange(n)
computed = np.log(computeSequence(np.asarray([2.25, -0.5]), 0, np.asarray([1/3, 1/12]), n))
exact = np.log(exactSolution(n))
ax.scatter(xticks, computed, label="Computed Sequence", marker="o", color="green")
ax.scatter(xticks, exact, label="Exact Solution", marker="x", color="blue")
    
ax.set_title("Semi-log plot of computed terms and exact terms of the sequence upto k="+str(n))
ax.set_xlabel("k")
ax.set_ylabel(r'$log(X_{k})$')
ax.legend(loc='center right')
fig.savefig("plot_upto_seq_"+str(n)+".png", bbox_inches="tight")
