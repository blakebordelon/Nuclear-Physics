import numpy as np
import matplotlib.pyplot as plt
import csv

def V_eff(r, l, j, is_neutron, params, spin_orbit=True):

	[Z,N,A] = params
	r_0 = 1.27 # fm
	mass_p = 931.5 # MeV / c^2
	mass_n = 939.56563  # MeV / c^2
	hbarc = 197.32697 # Mev fm

	centrifugal_const = 0
	if is_neutron:
		centrifugal_const = (hbarc**2)/(2*mass_n)
	else:
		centrifugal_const = (hbarc**2)/(2*mass_p)

	potential = 0
	V_0 = -51
	if is_neutron: # neutrons
		V_0 += 33 * (N-Z)/A
	else: # protons
		V_0 += -33 * (N-Z)/A

	V_so = -0.44 * V_0
	potential += V_0 * woods_saxon(r, params)
	if spin_orbit:
		potential += V_so * d_woods_saxon(r,params) * (r_0**2) * (j*(j+1) - l*(l+1) - 0.75)*0.5 / r
	potential += centrifugal_const* l * (l+1) / (r**2)
	if is_neutron == False:
		potential += coulomb(r,params)
		potential += 0
	return potential


def woods_saxon(r, params):
	r_0 = 1.27
	a = 0.67
	[Z,N,A] = params
	R = r_0 * np.power(A,1.0/3.0)
	return np.power( 1+ np.exp( (r-R)/a ), -1 )


def d_woods_saxon(r, params):
	r_0 = 1.27
	a = 0.67
	[Z,N,A] = params
	R = r_0 * np.power(A,1.0/3.0)

	ws = woods_saxon(r, params)
	return -(ws**2) * np.exp( (r-R)/a ) / a


def hamiltonian(rmesh, l, j, is_neutron, params):
	N = len(rmesh)
	H = np.zeros((N,N))
	mass_p = 938.5
	mass_n = 939.56563  # MeV / c^2
	hbarc = 197.32697 # MeV fm
	for i in range(N):
		H[i,i] += V_eff(rmesh[i], l, j, is_neutron, params)

	U = second_deriv(rmesh, l, j, params)
	if is_neutron:
		U = -(hbarc**2)/(2*mass_n) * U
	else:
		U = -(hbarc**2)/(2*mass_p) * U

	H += U
	return H


def second_deriv(rmesh, l, j, params):

	delta = rmesh[1]-rmesh[2]
	N = len(rmesh)
	r0 = rmesh[0]
	U = np.zeros((N,N))

	if l%2==0:
		U[0,0] = -3
		U[0,1] = 1
	else:
		U[0,0] = -1
		U[0,1] = 1

	for i in range(1,N-1):
		U[i,i-1] = 1
		U[i,i] = -2
		U[i,i+1] = 1

	U[N-1,N-1] = -2
	U[N-1,N-2] = 1
	
	U = U *np.power(delta, -2)
	return U

def get_eigenvals(rmesh, l, j, is_neutron, params):

	H = hamiltonian(rmesh, l, j, is_neutron, params)
	E, v = np.linalg.eig(H)
	return [E, v]

def coulomb(r, params):
	[Z,N,A] = params
	R = 1.27 * np.power(A, 1.0/3.0) # fm
	e_sqr = 1.4399764
	constant = e_sqr * Z
	potential = 0
	if r<R:
		potential += constant * (1.0/(2*R)) * (3 - np.power(r/R, 2))
	else:
		potential += constant / r

	return potential

def level_plot(my_levels, experiment, num_steps):

	xvals = np.linspace(1,2,num_steps)
	for i in range(len(my_levels)):
		plt.plot(xvals, my_levels[i] * np.ones(num_steps))
		plt.plot(xvals + 2* np.ones(num_steps), experiment[i] * np.ones(num_steps))
	plt.ylabel("Bound Level [MeV]")
	fig = plt.gcf()
	fig.savefig("proton_levels.pdf")
	plt.show()
	return

def plot_waves(rmesh, s_wave):
	
	for v in s_wave:
		plt.plot(rmesh, np.multiply(v, np.power(rmesh,-1)))
	
	plt.ylabel("R(r)")
	plt.xlabel("r [fm]")
	fig = plt.gcf()
	fig.savefig("swave_proton.pdf")
	plt.show()

	for v in s_wave:
		plt.plot(rmesh, v)
	
	plt.ylabel("u(r)")
	plt.xlabel("r [fm]")
	fig = plt.gcf()
	fig.savefig("swave_proton_u.pdf")
	plt.show()

	return

# BOMO values provided for comparison

Bomo_neutron = [ [-39.957, -29.476, -15.174, -1.262], [-35.752,-22.615,-7.290],
		[-36.119,-23.346,-8.193], [-30.685,-15.467, -0.646], 
		[-31.595,-16.931, -1.875], [-24.790, -8.046], [-26.497, -10.346],
		[-18.152,-0.725], [-20.908, -3.754], [-10.859], [-14.893],
		[-3.023], [-8.510],[], [-1.814]]

Bomo_proton = [[-32.334, -21.787, -7.028], [-28.523,-14.821],[-29.0587, -15.735],
				[-23.658, -7.511],[-24.902,-9.346],[-17.797],[-20.047, -2.695],
				[-11.0588],[-14.611],[-3.538],[-8.672],[],[-2.29]]

N = 600
rmesh = 16/N * np.linspace(1, N, num= N)
l = 0
j = 1/2

l_vals = np.linspace(0, 6, num = 7)

is_neutron = False

Bomo = []
if is_neutron:
	Bomo = Bomo_neutron
else:
	Bomo = Bomo_proton

params = [82,126,208]

V = np.zeros(len(rmesh))
V_no_so = np.zeros(len(rmesh))
for i in range(len(rmesh)):
	V[i] = V_eff(rmesh[i], l, j, is_neutron, params)
	V_no_so[i] = V_eff(rmesh[i], l, j, is_neutron, params, False)

plt.plot(rmesh, V)
plt.plot(rmesh, V_no_so)
plt.legend(["no spin orbit", "with spin orbit"])
plt.ylim([-50,50])
plt.ylabel("V [MeV]")
plt.xlabel("r [fm]")
plt.title("Effective Potential for f 7/2")
fig = plt.gcf()
fig.savefig("Vefff7_2.pdf")
plt.show()

All_Ebound = []
bomo_index = 0
all_diffs = []
s_wave = []
for l in l_vals:
	if l > 0:
		j_vals = np.linspace(-1/2, 1/2, num = 2) + l * np.ones(2)
	else:
		j_vals = np.ones(1)* 0.5

	jcount = 0
	for j in j_vals:
		E , v = get_eigenvals(rmesh, l, j, is_neutron, params)

		E_bound = [e for e in E if e < 0 and e > -50]
		vectors = [v[:,i] for i in range(len(E)) if E[i] < 0 and E[i]> -50]
		print("Ebound for l,j = %lf, %lf"% (l,j))
		E_bound.sort()
		print(E_bound)
		All_Ebound.append(E_bound)
		E_np = np.array(E_bound)
		exp = np.array(Bomo[bomo_index])
		bomo_index += 1
		if len(E_np) == len(exp):
			diff = np.multiply( (E_np - exp), np.power(exp,-1))
			#print(diff)
			all_diffs.append(diff)
		else:

			print(" lengths dont match ")
			print("length of my values: %d"  % len(E_np))
			print("length of bomo: %d" % len(exp))

		if l==0:
			s_wave = vectors

		jcount += 1
		n = 0
		




data = []
my_vals = []
exp_vals = []
all_diffs = np.abs(all_diffs)
for i in range(len(All_Ebound)):
	for j in range(len(All_Ebound[i])):
		current = [All_Ebound[i][j], Bomo[i][j], all_diffs[i][j]]
		data.append(current)
		my_vals.append(All_Ebound[i][j])
		exp_vals.append(Bomo[i][j])

plot_waves(rmesh, s_wave)
level_plot(my_vals, exp_vals, 10)

title = ""
if is_neutron:
	title = "neutron_levels.txt"
else:
	title = "proton_levels.txt"

with open(title, "w") as csvfile:
	writer = csv.writer(csvfile, delimiter = " ")
	writer.writerow(["My levels", "BOMO", "error"])
	for d in data:
		writer.writerow(np.sort(d))



