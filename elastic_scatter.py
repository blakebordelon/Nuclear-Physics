import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt


def fermi_function(r_vals, fermi_params):

	[R, a] = fermi_params
	num_steps = len(r_vals)
	normalize = 0
	dr = r_vals[1] - r_vals[0]
	for i in range(num_steps):
		normalize += 1/(1+np.exp((r_vals[i] - R)/a)) * np.power(r_vals[i],2) * dr

	rho_0 = 82 / (4 * math.pi * normalize)

	density = np.zeros(len(r_vals))
	for i in range(len(r_vals)):
		density[i] = rho_0 / (1 + np.exp((r_vals[i] - R)/a) )

	return density


def square_well_solution(r_vals, well_params):
	[R, C, k, kappa] = well_params
	wave_function = np.zeros(len(r_vals))
	frac_coeff = np.exp(- kappa * R) / np.sin(k*R)

	norm = np.power(frac_coeff,2) * (R + np.sin(2 *k* R)) + np.power(kappa,-1)*np.exp(-2*kappa*R)

	B = np.sqrt(2 / norm)
	A = frac_coeff * B 

	for i in range(len(r_vals)):
		r = r_vals[i]
		if r<R:
			wave_function[i] = A * np.sin(k * r)

		else:
			wave_function[i] = B * np.exp(- kappa * r)

	return wave_function

def square_well_potential(r_vals, well_params):
	[R, C, k, kappa] = well_params
	pot = np.zeros(len(r_vals))
	for i in range(len(r_vals)):
		r = r_vals[i]
		if r < R:
			pot[i] = 0
		else:
			pot[i] = 1
	return pot

def mott_dcs(q_vals, e_params):
	[alpha, m, hbarc, Z] = e_params
	dcs = np.zeros(len(q_vals))
	for i in range(len(q_vals)):
		dcs[i] = np.power(2 * m * alpha * Z /( hbarc *  q_vals[i] * q_vals[i] ), 2)
	return dcs


def sqr_form_factor(charge_density, r_vals, q_vals):

	form_factor = np.zeros(len(q_vals))
	delta = r_vals[1] - r_vals[2]
	for q_i in range(len(q_vals)):
		q = q_vals[q_i]
		for r in r_vals:
			form_factor[q_i] += np.sin(q*r) / (q) * 4 * math.pi * r * delta

	sqrd_form_factor = np.power( np.abs(form_factor), 2)

	return sqrd_form_factor






R_max = 12		# fm
Q_max = 5		# fm
rho_0 = .063	# e/fm^3
R = 6.6			# fm
a = 0.4			# fm
alpha = 1/137	# dimensionless 
m = .511	 	# MeV / c^2
hbarc = 197.32	# MeV fm
Z = 82		 	# lead nucleus


R = 2
C = 3
k = 1.18
kappa = 1.60



fermi_params = [R, a]
e_params = [alpha, m, hbarc, Z]
sqr_params = [R,C,k, kappa]

r_len = 1000
q_len = 1000

r_vals = np.linspace(0, R_max, r_len)
q_vals = np.linspace(1e-6, Q_max, q_len)


wf = square_well_solution(r_vals, sqr_params)
pot = square_well_potential(r_vals, sqr_params)
plt.plot(r_vals, np.multiply(np.power(wf,2), np.power(r_vals,-2)))
plt.plot(r_vals, wf)
plt.plot(r_vals, pot)
plt.title("Square Well Solution")
plt.xlabel('r [fm]')
plt.legend(['charge density', 'wave_function' , 'potential'])
fig = plt.gcf()
fig.savefig("SquareWell.png")
plt.show()

form_he = sqr_form_factor(np.multiply( np.power(r_vals, -2), np.power(wf,2)), r_vals, q_vals)
dcs_he = np.multiply( mott_dcs(q_vals, e_params), form_he)


plt.plot(q_vals, np.log10(form_he))
plt.title("Form Factor for He-4")
plt.xlabel("q [fm^-1]")
plt.xlim([0.5,5])
plt.ylabel("F(q)^2")
fig = plt.gcf()
fig.savefig("HeFormFactor.png")
plt.show()

plt.plot(q_vals, np.log10(dcs_he))
plt.title("Elastic Electron Scattering for He-4")
plt.xlim([0.5,5])
plt.ylabel("DCS")
plt.xlabel("q [ fm^-1]")
fig = plt.gcf()
fig.savefig("HeDCS.png")
plt.show()


density = fermi_function(r_vals, fermi_params)
mott = mott_dcs(q_vals, e_params)
form_factor = sqr_form_factor(density, r_vals, q_vals)

dcs_final = np.multiply(mott, form_factor)

plt.title("Generic Lead Nucleus")
plt.plot(r_vals, density)
plt.xlabel('r [fm]')
plt.ylabel('density [e/fm^3]')
fig = plt.gcf()
fig.savefig("LeadDensity.png")
plt.show()


plt.xlim([0.5,5])
plt.title("Form Factor for Lead Nucleus")
plt.plot(q_vals, np.log10(form_factor))
plt.xlabel('q [fm^-1]')
plt.ylabel('|F(q)|^2')
fig = plt.gcf()
fig.savefig("LeadFormFactor.png")
plt.show()


plt.xlim([0.5, 5])
plt.ylim([-12,1])
plt.title("Elastic Electron Differential Cross Section")
plt.xlabel('q [fm^-1]')
plt.ylabel('DCS [mb/sr]')
plt.plot(q_vals, np.log10(dcs_final))
fig = plt.gcf()
fig.savefig("LeadCrossSection.png")
plt.show()