# -*- coding: utf-8 -*-

"""Plesio-Geostrophy equations"""


import sympy
from sympy import diff
from sympy import Derivative as diff_u
from sympy_supp import vector_calculus_3d as v3d


"""Initialization"""

# ### Variables

# In[64]:


x, y, z, t = sympy.symbols("x, y, z, t", real=True)
s, p, r, theta = sympy.symbols(r"s, \phi, r, \theta", positive=True)
n, m = sympy.symbols("n, m", integer=True)
H = sympy.Function("H")(s)
H_s = sympy.sqrt(1 - s**2)

cyl_op = v3d.CylindricalCoordinates((s, p, z))
sph_op = v3d.SphericalCoordinates((r, theta, p))


# ### Complete fields

# In[3]:


Psi = sympy.Function(r"\Psi")(s, p, t)

Mss = sympy.Function(r"\overline{M_{ss}}")(s, p, t)
Msp = sympy.Function(r"\overline{M_{s\phi}}")(s, p, t)
Mpp = sympy.Function(r"\overline{M_{\phi\phi}}")(s, p, t)

Msz = sympy.Function(r"\widetilde{M_{sz}}")(s, p, t)
Mpz = sympy.Function(r"\widetilde{M_{\phi z}}")(s, p, t)
zMss = sympy.Function(r"\widetilde{zM_{ss}}")(s, p, t)
zMpp = sympy.Function(r"\widetilde{zM_{\phi\phi}}")(s, p, t)
zMsp = sympy.Function(r"\widetilde{zM_{s\phi}}")(s, p, t)

Bs_e = sympy.Function(r"B_{es}")(s, p, t)
Bp_e = sympy.Function(r"B_{e\phi}")(s, p, t)
Bz_e = sympy.Function(r"B_{ez}")(s, p, t)

dBs_dz_e = sympy.Function(r"B_{es, z}")(s, p, t)
dBp_dz_e = sympy.Function(r"B_{e\phi, z}")(s, p, t)

Br = sympy.Function(r"B_r")(theta, p, t)

list_fields = [Psi, Mss, Mpp, Msp, Msz, Mpz, zMss, zMpp, zMsp, Bs_e, Bp_e, Bz_e, dBs_dz_e, dBp_dz_e, Br]


# ### Boundary terms

# In[4]:


Bs_p = sympy.Function(r"B_s^+")(s, p, t)
Bp_p = sympy.Function(r"B_\phi^+")(s, p, t)
Bz_p = sympy.Function(r"B_z^+")(s, p, t)
Bs_m = sympy.Function(r"B_s^-")(s, p, t)
Bp_m = sympy.Function(r"B_\phi^-")(s, p, t)
Bz_m = sympy.Function(r"B_z^-")(s, p, t)

Msz_p_expr = Bs_p*Bz_p
Mpz_p_expr = Bp_p*Bz_p
Mss_p_expr = Bs_p*Bs_p
Msp_p_expr = Bs_p*Bp_p
Mzz_p_expr = Bz_p*Bz_p
Msz_m_expr = Bs_m*Bz_m
Mpz_m_expr = Bp_m*Bz_m
Mss_m_expr = Bs_m*Bs_m
Msp_m_expr = Bs_m*Bp_m
Mzz_m_expr = Bz_m*Bz_m

list_boundaries = [Bs_p, Bp_p, Bz_p, Bs_m, Bp_m, Bz_m]


# ### Auxiliary fields

# In[35]:


us = sympy.Function(r"u_s")(s, p, t)
up = sympy.Function(r"u_\phi")(s, p, t)
uz = sympy.Function(r"u_z")(s, p, z, t)

us_psi = 1/(s*H)*diff(Psi, p)
up_psi = -1/H*diff(Psi, s)
uz_psi = z/H*diff(H, s)*us_psi


# ---
# 
# ## Plesio-Geostrophy Equations
# 
# Governing equations for the PG model

# ### Vorticity equation
# 
# For now I omit the viscous dissipation terms

# In[25]:


Ls_sym = sympy.Function(r"\overline{L_s}")(s, p, t)
Lp_sym = sympy.Function(r"\overline{L_\phi}")(s, p, t)
Lz_asym = sympy.Function(r"\widetilde{L_z}")(s, p, t)
Le_p = sympy.Function(r"L_{e\phi}")(s, p, t)

fs_sym = sympy.Function(r"\overline{f_s}")(s, p, t)
fp_sym = sympy.Function(r"\overline{f_\phi}")(s, p, t)
fz_asym = sympy.Function(r"\widetilde{f_z}")(s, p, t)
fe_p = sympy.Function(r"f_{e\phi}")(s, p, t)

# vorticity_var = -2*cyl_op.laplacian(diff(Psi, t)) + diff(H, s)*(2/H*diff(Psi, t, s) + 1/s/H*diff(Psi, t, (p, 2)))
# vorticity_forcing = diff(H, s)*4/s/H*diff(Psi, p) - diff(H, s)*(2*fe_p + 1/s*diff(fz_asym, p)) + cyl_op.curl((fs_sym, fp_sym, 0))[2]
vorticity_var = diff(s/H*diff(Psi, t, s), s) + (1/(s*H) - 1/(2*H**2)*diff(H, s))*diff(Psi, t, (p, 2))
vorticity_forcing = 2/H**2*diff(H, s)*diff(Psi, p) - diff(H, s)*(s/H*fe_p + 1/(2*H)*diff(fz_asym, p)) + s/(2*H)*cyl_op.curl((fs_sym, fp_sym, 0))[2]


# ### Lorentz force

# In[6]:


Ls_sym_expr = 1/s*diff(s*Mss, s) + 1/s*diff(Msp, p) - Mpp/s + (Msz_p_expr - Msz_m_expr) + s/H*(Mss_p_expr + Mss_m_expr)
Lp_sym_expr = 1/s*diff(s*Msp, s) + 1/s*diff(Mpp, p) + Msp/s + (Mpz_p_expr - Mpz_m_expr) + s/H*(Msp_p_expr + Msp_m_expr)
Lz_asym_expr = 1/s*diff(s*Msz, s) + 1/s*diff(Mpz, p) + (Mzz_p_expr + Mzz_m_expr - 2*Bz_e*Bz_e) + s/H*(Msz_p_expr - Msz_m_expr)
Le_p_expr = Bs_e*diff(Bp_e, s) + 1/s*Bp_e*diff(Bp_e, p) + Bz_e*dBp_dz_e + 1/s*Bs_e*Bp_e

# ### Alternative form in Cartesian components
# 
# Only moment tensor components are cartesian; the fields are still defined in cylindrical coordinates.

# In[ ]:


Mxx = sympy.Function(r"\overline{M_{xx}}")(s, p, t)
Mxy = sympy.Function(r"\overline{M_{xy}}")(s, p, t)
Myy = sympy.Function(r"\overline{M_{yy}}")(s, p, t)

Mxz = sympy.Function(r"\widetilde{M_{xz}}")(s, p, t)
Myz = sympy.Function(r"\widetilde{M_{yz}}")(s, p, t)
zMxx = sympy.Function(r"\widetilde{zM_{xx}}")(s, p, t)
zMyy = sympy.Function(r"\widetilde{zM_{yy}}")(s, p, t)
zMxy = sympy.Function(r"\widetilde{zM_{xy}}")(s, p, t)

rank2_tensor_transform = {
    Mss: sympy.cos(p)**2*Mxx + sympy.sin(p)**2*Myy + 2*sympy.sin(p)*sympy.cos(p)*Mxy,
    Mpp: sympy.sin(p)**2*Mxx + sympy.cos(p)**2*Myy - 2*sympy.sin(p)*sympy.cos(p)*Mxy,
    Msp: sympy.cos(p)*sympy.sin(p)*(Myy - Mxx) + (sympy.cos(p)**2 - sympy.sin(p)**2)*Mxy,
    Msz: sympy.cos(p)*Mxz + sympy.sin(p)*Myz,
    Mpz: -sympy.sin(p)*Mxz + sympy.cos(p)*Myz,
    zMss: sympy.cos(p)**2*zMxx + sympy.sin(p)**2*zMyy + 2*sympy.sin(p)*sympy.cos(p)*zMxy,
    zMpp: sympy.sin(p)**2*zMxx + sympy.cos(p)**2*zMyy - 2*sympy.sin(p)*sympy.cos(p)*zMxy,
    zMsp: sympy.cos(p)*sympy.sin(p)*(zMyy - zMxx) + (sympy.cos(p)**2 - sympy.sin(p)**2)*zMxy,
}

Ls_sym_expr_cart = Ls_sym_expr.subs(rank2_tensor_transform)
Lp_sym_expr_cart = Ls_sym_expr.subs(rank2_tensor_transform)
Lz_asym_expr_cart = Ls_sym_expr.subs(rank2_tensor_transform)


# ### Induction equation - the magnetic moments

# In[36]:


v_e = (us, up, 0)

evo_Mss = cyl_op.grad(Mss/H, evaluate=False)
evo_Mss = -H*v3d.dot(v_e, evo_Mss) + 2*diff_u(us, s)*Mss + 2/s*diff_u(us, p)*Msp

evo_Mpp = cyl_op.grad(H*Mpp, evaluate=False)
evo_Mpp = -1/H*v3d.dot(v_e, evo_Mpp) - 2*diff_u(us, s)*Mpp + 2*s*diff_u(up/s, s)*Msp

evo_Msp = cyl_op.grad(Msp, evaluate=False)
evo_Msp = -v3d.dot(v_e, evo_Msp) + s*diff_u(up/s, s)*Mss + 1/s*diff_u(us, p)*Mpp

evo_Msz = cyl_op.grad(Msz, evaluate=False)
evo_Msz = -v3d.dot(v_e, evo_Msz) + (diff_u(us, s) + 2*diff_u(uz, z))*Msz + 1/s*diff_u(us, p)*Mpz + diff_u(us/H*diff_u(H, s), s)*zMss + 1/(s*H)*diff_u(H, s)*diff_u(us, p)*zMsp

evo_Mpz = cyl_op.grad(Mpz, evaluate=False)
evo_Mpz = -v3d.dot(v_e, evo_Mpz) + (diff(uz, z) - diff(us, s))*Mpz + diff_u(us/H*diff(H, s), s)*zMsp + 1/(s*H)*diff(H, s)*diff(us, p)*zMpp

evo_zMss = cyl_op.grad(zMss, evaluate=False)
evo_zMss = -v3d.dot(v_e, evo_zMss) + 2*(diff(us, s) + diff(uz, z))*zMss + 2/s*diff(us, p)*zMsp

evo_zMpp = cyl_op.grad(zMpp, evaluate=False)
evo_zMpp = -v3d.dot(v_e, evo_zMpp) - 2*diff(us, s)*zMpp + 2*s*diff_u(up/s, s)*zMsp

evo_zMsp = cyl_op.grad(zMsp, evaluate=False)
evo_zMsp = -v3d.dot(v_e, evo_zMsp) + diff(uz, z)*zMsp + s*diff_u(up/s, s)*zMss + 1/s*diff(us, p)*zMpp


# ### Induction: magnetic field in the equatorial plane

# In[37]:


evo_Bs_e = Bs_e*diff(us, s) + 1/s*Bp_e*diff(us, p) - us*diff(Bs_e, s) - 1/s*up*diff(Bs_e, p)
evo_Bp_e = Bs_e*diff(up, s) + 1/s*Bp_e*diff(up, p) - us*diff(Bp_e, s) - 1/s*up*diff(Bp_e, p) + (Bp_e*us - up*Bs_e)/s
evo_Bz_e = -us*diff(Bz_e, s) - 1/s*up*diff(Bz_e, p) + diff(uz, z)*Bz_e
evo_dBs_dz_e = dBs_dz_e*diff(us, s) + 1/s*dBp_dz_e*diff(us, p) - us*diff(dBs_dz_e, s) - 1/s*up*diff(dBs_dz_e, p) - diff(uz, z)*dBs_dz_e
evo_dBp_dz_e = dBs_dz_e*diff(up, s) + 1/s*dBp_dz_e*diff(up, p) - us*diff(dBp_dz_e, s) - 1/s*up*diff(dBp_dz_e, p) + (dBp_dz_e*us - up*dBs_dz_e)/s - diff(uz, z)*dBp_dz_e


# ### Induction: boundary stirring

# In[65]:


ur = sympy.Function(r"u_r")(r, theta, p)
ut = sympy.Function(r"u_\theta")(r, theta, p)
up_sph = sympy.Function(r"u_\phi")(r, theta, p)

evo_Br = -sph_op.surface_div((Br*ut, Br*up_sph), evaluate=False)


# ## Linearized equations
# 
# Introduce a small quantity $\epsilon$

# In[7]:


eps = sympy.Symbol("\epsilon")


# ### Unperturbed fields

# In[8]:


Psi_0 = sympy.Function(r"\Psi^0")(s, p, t)

Mss_0 = sympy.Function(r"\overline{M_{ss}}^0")(s, p, t)
Msp_0 = sympy.Function(r"\overline{M_{s\phi}}^0")(s, p, t)
Mpp_0 = sympy.Function(r"\overline{M_{\phi\phi}}^0")(s, p, t)

Msz_0 = sympy.Function(r"\widetilde{M_{sz}}^0")(s, p, t)
Mpz_0 = sympy.Function(r"\widetilde{M_{\phi z}}^0")(s, p, t)
zMss_0 = sympy.Function(r"\widetilde{zM_{ss}}^0")(s, p, t)
zMpp_0 = sympy.Function(r"\widetilde{zM_{\phi\phi}}^0")(s, p, t)
zMsp_0 = sympy.Function(r"\widetilde{zM_{s\phi}}^0")(s, p, t)

Bs_e_0 = sympy.Function(r"B_{es}^0")(s, p, t)
Bp_e_0 = sympy.Function(r"B_{e\phi}^0")(s, p, t)
Bz_e_0 = sympy.Function(r"B_{ez}^0")(s, p, t)

dBs_dz_e_0 = sympy.Function(r"B_{es, z}^0")(s, p, t)
dBp_dz_e_0 = sympy.Function(r"B_{e\phi, z}^0")(s, p, t)

Br_0 = sympy.Function(r"B_r^0")(theta, p, t)

list_bg_fields = [Psi_0, Mss_0, Mpp_0, Msp_0, Msz_0, Mpz_0, zMss_0, zMpp_0, zMsp_0, Bs_e_0, Bp_e_0, Bz_e_0, dBs_dz_e_0, dBp_dz_e_0, Br_0]

# Boundary terms

Bs_p_0 = sympy.Function(r"B_s^{0+}")(s, p, t)
Bp_p_0 = sympy.Function(r"B_\phi^{0+}")(s, p, t)
Bz_p_0 = sympy.Function(r"B_z^{0+}")(s, p, t)
Bs_m_0 = sympy.Function(r"B_s^{0-}")(s, p, t)
Bp_m_0 = sympy.Function(r"B_\phi^{0-}")(s, p, t)
Bz_m_0 = sympy.Function(r"B_z^{0-}")(s, p, t)

list_bg_boundaries = [Bs_p_0, Bp_p_0, Bz_p_0, Bs_m_0, Bp_m_0, Bz_m_0]


# ### Perturbation

# In[9]:


psi = sympy.Function(r"\psi")(s, p, t)

mss = sympy.Function(r"\overline{m_{ss}}")(s, p, t)
msp = sympy.Function(r"\overline{m_{s\phi}}")(s, p, t)
mpp = sympy.Function(r"\overline{m_{\phi\phi}}")(s, p, t)

msz = sympy.Function(r"\widetilde{m_{sz}}")(s, p, t)
mpz = sympy.Function(r"\widetilde{m_{\phi z}}")(s, p, t)
zmss = sympy.Function(r"\widetilde{zm_{ss}}")(s, p, t)
zmpp = sympy.Function(r"\widetilde{zm_{\phi\phi}}")(s, p, t)
zmsp = sympy.Function(r"\widetilde{zm_{s\phi}}")(s, p, t)

bs_e = sympy.Function(r"b_{es}")(s, p, t)
bp_e = sympy.Function(r"b_{e\phi}")(s, p, t)
bz_e = sympy.Function(r"b_{ez}")(s, p, t)

dbs_dz_e = sympy.Function(r"b_{es, z}")(s, p, t)
dbp_dz_e = sympy.Function(r"b_{e\phi, z}")(s, p, t)

br = sympy.Function(r"b_r")(theta, p, t)

list_perturb_fields = [psi, mss, mpp, msp, msz, mpz, zmss, zmpp, zmsp, bs_e, bp_e, bz_e, dbs_dz_e, dbp_dz_e, br]

# Boundary terms

bs_p = sympy.Function(r"b_s^+")(s, p, t)
bp_p = sympy.Function(r"b_\phi^+")(s, p, t)
bz_p = sympy.Function(r"b_z^+")(s, p, t)
bs_m = sympy.Function(r"b_s^-")(s, p, t)
bp_m = sympy.Function(r"b_\phi^-")(s, p, t)
bz_m = sympy.Function(r"b_z^-")(s, p, t)

list_perturb_boundaries = [bs_p, bp_p, bz_p, bs_m, bp_m, bz_m]


# First we define the substitutions / expansions

# In[10]:


linearization_subs_map = {
    list_fields[idx_field]: list_bg_fields[idx_field] + eps*list_perturb_fields[idx_field] for idx_field in range(len(list_fields))
}
linearization_subs_map.update({
    list_boundaries[idx_bound]: list_bg_boundaries[idx_bound] + eps*list_perturb_boundaries[idx_bound] for idx_bound in range(len(list_boundaries))
})


# ### Linearized vorticity equation

# In[22]:


vorticity_var_perturbed = vorticity_var.subs(linearization_subs_map)
vorticity_var_lin = vorticity_var_perturbed.simplify().expand().coeff(eps, 1)

vorticity_forcing_perturbed = vorticity_forcing.subs(linearization_subs_map)
vorticity_forcing_lin = sympy.collect(vorticity_forcing_perturbed.simplify().expand(), eps).coeff(eps, 1)


# ### Linearized Lorentz force
# 
# Linearized in terms of magnetic fields (for $L_{e\phi}$) or in terms of magnetic moments (for integrated forces).

# Lorentz force in the equatorial plane is quadratic in the magnetic field components in the equatorial plane. Linearized form involves cross terms between background magnetic field and perturbational fields.
# 
# Linearized form of $L_{e\phi}$:

# In[47]:


Le_p_perturbed = Le_p_expr.subs(linearization_subs_map)
Le_p_perturbed = sympy.collect(Le_p_perturbed.simplify().expand(), eps)

Le_p_bg = Le_p_perturbed.coeff(eps, 0)
Le_p_lin = Le_p_perturbed.coeff(eps, 1)


# For the integrated quantities, the Lorentz force IS a linear function of magnetic moments. No linearization required. However, the boundary terms and the equatorial terms are quadratic in magnetic fields. These terms need to be linearized.
# 
# Linearized form for $\overline{L_s}$

# In[48]:


Ls_sym_perturbed = Ls_sym_expr.subs(linearization_subs_map)
Ls_sym_perturbed = sympy.collect(Ls_sym_perturbed.simplify().expand(), eps)

Ls_sym_bg = Ls_sym_perturbed.coeff(eps, 0)
Ls_sym_lin = Ls_sym_perturbed.coeff(eps, 1)


# Linearized form for $\overline{L_\phi}$:

# In[49]:


Lp_sym_perturbed = Lp_sym_expr.subs(linearization_subs_map)
Lp_sym_perturbed = sympy.collect(Lp_sym_perturbed.simplify().expand(), eps)

Lp_sym_bg = Lp_sym_perturbed.coeff(eps, 0)
Lp_sym_lin = Lp_sym_perturbed.coeff(eps, 1)



# Linearized form for $\widetilde{L_z}$:

# In[50]:


Lz_asym_perturbed = Lz_asym_expr.subs(linearization_subs_map)
Lz_asym_perturbed = sympy.collect(Lz_asym_perturbed.simplify().expand(), eps)

Lz_asym_bg = Lz_asym_perturbed.coeff(eps, 0)
Lz_asym_lin = Lz_asym_perturbed.coeff(eps, 1)



# Curl of horizontal components $\nabla \times \mathbf{L}_e$:

# In[62]:


curl_L = cyl_op.curl((Ls_sym_lin, Lp_sym_lin, 0))[2]

curl_L.simplify().expand()


# In[59]:


curl_L_malkus = curl_L.subs({
    Bs_p_0: 0,
    Bs_m_0: 0,
    Bz_p_0: 0,
    Bz_m_0: 0,
    Bp_p_0: s,
    Bp_m_0: s
})
curl_L_malkus.simplify().expand()


# In[ ]:


