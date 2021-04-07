#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


# ## Part A

# In[2]:


def lup(A):
    U = A.copy()
    m = A.shape[0]
    L = np.eye(m)
    P = np.eye(m)
    for k in range(0, m-1):
        i = k + np.argmax(abs(U[k:, k]))
        temp = U[k, k:].copy()
        U[k, k:] = U[i, k:]
        U[i, k:] = temp
        temp = L[k, :k].copy()
        L[k, :k] = L[i, :k]
        L[i, :k] = temp
        temp = P[k, :].copy()
        P[k, :] = P[i, :]
        P[i, :] = temp
        for j in range(k+1, m):
            L[j, k] = U[j, k]/U[k, k]
            U[j, k:] -= L[j, k]*U[k, k:]
    return P, L, U


# In[3]:


def solveLup(P, L, U, b):
    b = P @ b
    m = L.shape[0]
    w = np.zeros(m)
    for i in range(m):
        temp = b[i]
        for j in range(i):
            temp -= L[i, j]*w[j]
        w[i] = temp/L[i,i]
    
    x = np.zeros(m)
    for i in reversed(np.arange(0, m)):
        temp = w[i]
        for j in range(i+1, m):
            temp -= U[i,j]*x[j]
        x[i] = temp/U[i,i]
    
    return x.reshape(-1,1)


# In[4]:


def instabilityMatrix(m):
    x = np.eye(m)
    x[:, m-1] = 1
    for i in range(m):
        x[i, :i] = -1
    return x


# In[5]:


A = np.asarray([[2,1,1,0], [4,3,3,1], [8,7,9,5], [6,7,9,8]], dtype=np.float64)


# In[6]:


P, L, U = lup(A)


# In[7]:


b = np.asarray([[3], [5], [10], [17]])


# In[8]:


print(solveLup(P, L, U, b))


# In[9]:


print(instabilityMatrix(5))


# ## Part C

# In[10]:


def rookPivot(U, i, j, k):
    ip = k + np.argmax(abs(U[k:, j]))
    jp = k + np.argmax(abs(U[ip, k:]))
    return ((i, j) if ((ip==i) and (jp==j)) else rookPivot(U, ip, jp, k))


# In[11]:


def lupq(A):
    U = A.copy()
    m = A.shape[0]
    L = np.eye(m)
    P = np.eye(m)
    Q = np.eye(m)
    for k in range(0, m-1):
        i, j = rookPivot(U, k, k, k)
        temp = U[k, k:].copy()
        U[k, k:] = U[i, k:]
        U[i, k:] = temp
        temp = U[:, k].copy()
        U[:, k] = U[:, j]
        U[:, j] = temp
        temp = L[k, :k].copy()
        L[k, :k] = L[i, :k]
        L[i, :k] = temp
        temp = P[k, :].copy()
        P[k, :] = P[i, :]
        P[i, :] = temp
        temp = Q[:, k].copy()
        Q[:, k] = Q[:, j]
        Q[:, j] = temp
        for n in range(k+1, m):
            L[n, k] = U[n, k]/U[k, k]
            U[n, k:] -= L[n, k]*U[k, k:]
    return P, Q, L, U


# In[12]:


def solveLupq(P, Q, L, U, b):
    b = P @ b
    m = L.shape[0]
    w = np.zeros(m)
    for i in range(m):
        temp = b[i]
        for j in range(i):
            temp -= L[i, j]*w[j]
        w[i] = temp/L[i,i]
    
    y = np.zeros(m)
    for i in reversed(np.arange(0, m)):
        temp = w[i]
        for j in range(i+1, m):
            temp -= U[i,j]*y[j]
        y[i] = temp/U[i,i]
    
    x = Q @ y.reshape(-1, 1)
    return x


# In[13]:


P, Q, L, U = lupq(A)


# In[14]:


print(solveLupq(P,Q, L, U, b))


# In[15]:


rho1 = []
rho2 = []
back1 = []
back2 = []
n = 60
xticks = np.arange(1, n+1)
for i in xticks:
    A = instabilityMatrix(i)
    b = np.random.randn(i, 1)
    P1, L1, U1 = lup(A)
    P2, Q2, L2, U2 = lupq(A)
    x1 = (A @ solveLup(P1, L1, U1, b)).flatten()
    x2 = (A @ solveLupq(P2, Q2, L2, U2, b)).flatten()
    b = b.flatten()
    rho1.append(np.log(abs(U1).max() / abs(A).max()))
    rho2.append(np.log(abs(U2).max() / abs(A).max()))
    back1.append(np.log(np.linalg.norm(x1-b, 2) / np.linalg.norm(b, 2)))
    back2.append(np.log(np.linalg.norm(x2-b, 2) / np.linalg.norm(b, 2)))
    
fig, ax = plt.subplots(1, 1, figsize =(16, 7))
ax.scatter(xticks, rho1, label="Partial Pivoting", marker="o", color="green")
ax.scatter(xticks, rho2, label="Rook Pivoting", marker="x", color="blue")
    
ax.set_title("Semi-log plot of growth factor rho as function of m upto m="+str(n))
ax.set_xlabel("m")
ax.set_ylabel(r'$log(rho_{m})$')
ax.legend(loc='center right')
fig.savefig("plot_rho_upto_m_"+str(n)+".png", bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize =(16, 7))
ax.scatter(xticks, back1, label="Partial Pivoting", marker="o", color="green")
ax.scatter(xticks, back2, label="Rook Pivoting", marker="x", color="blue")
    
ax.set_title("Semi-log Plot of relative backward error as function of m upto m="+str(n))
ax.set_xlabel("m")
ax.set_ylabel(r'$log(relative\;backward\;error)$')
ax.legend(loc='upper left')
fig.savefig("plot_rbe_upto_m_"+str(n)+".png", bbox_inches="tight")


# In[ ]:




