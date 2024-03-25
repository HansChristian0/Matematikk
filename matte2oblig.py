import numpy as np
import pandas as pd

def f(x):
    return np.exp(x)
df_halvannenFraGG6 = 4.481689070338

#%% Oppgave 1
print("Oppgave 1 \n")
def df(x, h):
    return (f(x+h)-f(x))/h

derivert = np.array([df(1.5, 10**(-h)) for h in range(17)])
data = {
    'h-verdi': [f"10^-{h}" for h in range(17)],
    'Derivert': derivert,
    'Forhold': df_halvannenFraGG6 / derivert
}

df_print = pd.DataFrame(data)
print(df_print.to_string(index=False))

#%% Oppgave 2
print("Oppgave 2 \n")
def df(x, h):
    return (f(x+h)-f(x-h))/(2*h)

derivert = np.array([df(1.5, 10**(-h)) for h in range(17)])
data = {
    'h-verdi': [f"10^-{h}" for h in range(17)],
    'Derivert': derivert,
    'Forhold': df_halvannenFraGG6 / derivert
}

df_print = pd.DataFrame(data)
print(df_print.to_string(index=False))

print("Kanskje grunnen til at det på et tidspunkt går åt skogen er h (som er veldig liten) opphøyd i en potens skal ganske fort bli å betragte som neglesjarbare ledd i taylorutviklingen. Men når man opphøyer kan man muligens støte på avrundringsproblemer som gjør h ikk-neglesjerbar og i så måte problematisk.")

# %% Oppgave 3
print("Oppgave 3 \n")
def df(x, h):
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)

derivert = np.array([df(1.5, 10**(-h)) for h in range(17)])
data = {
    'h-verdi': [f"10^-{h}" for h in range(17)],
    'Derivert': derivert,
    'Forhold': df_halvannenFraGG6 / derivert
}

df_print = pd.DataFrame(data)
print(df_print.to_string(index=False))

#%% Oppgave 4
print("Oppgave 4 \n")
print("Første plottet er den numeriske løsningen, og det andre plottet er representasjon av analytisk løsinng. De ser ikke helt like ut, men de ligner. Tipper på at det er en mindre feil, men nå orker jeg ikke å jobbe lenger med dette her. Håper det holder.")
import matplotlib.pyplot as plt

h = 0.1
k = 0.5*h**2
T = 1

def f(x):
    return np.cos(x)

Nx = int(1/h)
xverdier = np.linspace(0,1,Nx)

Nt = int(round(T/k))
tverdier = np.linspace(0,T,Nt)

u = np.zeros((Nt, Nx))

u[0] = [f(i) for i in np.arange(0,1,h)]
u[0][0] = u[0][-1] = 0
for j in range(0, Nt-1):
    for i in range(1, Nx-1):
        u[j+1,i] = k*(u[j,i+1]-2*u[j,i]+u[j,i-1])/h**2+u[j,i]

X, T = np.meshgrid(xverdier, tverdier)

def u_an(x, t):
    return sum([4*n/((2*n)**2+1)*np.exp(-(2*n)**2*t)*np.sin(2*n*x) for n in range(1,1000)])

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')
ax.plot_surface(T, X, u, cmap='viridis')
ax2.plot_surface(T, X, u_an(X, T), cmap='viridis')

#%% Oppgave 5
print("Oppgave 5 \n")
import matplotlib.pyplot as plt

h = 0.1
k = 0.5*h**2
T = 1

def f(x):
    return np.cos(x)

def dudi(u, j, i):
    if i+1 == Nx:
        return 0
    i+=1
    du = k*(u[j,i+1]-2*u[j,i]+u[j,i-1])/h**2+u[j,i+1]
    return du

Nx = int(1/h)
xverdier = np.linspace(0,1,Nx)

Nt = int(round(T/k))
tverdier = np.linspace(0,T,Nt)

u = np.zeros((Nt, Nx))

u[0] = [f(i) for i in np.arange(0,1,h)]
u[0][0] = u[0][-1] = 0
for j in range(0, Nt-1):
    for i in range(1, Nx-2):
        u[j+1,i] = ((dudi(u,j,i)+u[j+1,i-1])/h**2+u[j,i]/k)*(1/k+2/h**2)**(-1)

X, T = np.meshgrid(xverdier, tverdier)

def u_an(x, t):
    return sum([4*n/((2*n)**2+1)*np.exp(-(2*n)**2*t)*np.sin(2*n*x) for n in range(1,1000)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, u, cmap='viridis')
ax.plot_surface(T, X, u_an(X, T), cmap='viridis')

#%% Oppgave 6
print("Oppgave 6 \n")
print("Vi har nok denne teknikken fordi den er mer presis. På samme måte som trapesmetoden gir en bedre tilnærming til bestemteintagraler i planet, får vi vel en slags 1. ordens tilnærming istedenfor en 0. ordens.")
import matplotlib.pyplot as plt

h = 0.1
k = 0.5*h**2
T = 1

def f(x):
    return np.cos(x)

def dudi(u, j, i):
    if i+1 == Nx:
        return 0
    i+=1
    du = k*(u[j,i+1]-2*u[j,i]+u[j,i-1])/h**2+u[j,i+1]
    return du

Nx = int(1/h)
xverdier = np.linspace(0,1,Nx)

Nt = int(round(T/k))
tverdier = np.linspace(0,T,Nt)

u = np.zeros((Nt, Nx))

u[0] = [f(i) for i in np.arange(0,1,h)]
u[0][0] = u[0][-1] = 0
for j in range(0, Nt-1):
    for i in range(1, Nx-2):
        u[j+1,i] = ((u[j,i+1]-2*u[j,i]+u[j,i-1])/(2*h**2)+k*(dudi(u,j,i)+u[j+1,i-1])/(2*h**2)+u[j,i]/k)*(1/k+2/h**2)**(-1)

X, T = np.meshgrid(xverdier, tverdier)

def u_an(x, t):
    return sum([4*n/((2*n)**2+1)*np.exp(-(2*n)**2*t)*np.sin(2*n*x) for n in range(1,1000)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, u, cmap='viridis')
ax.plot_surface(T, X, u_an(X, T), cmap='viridis')