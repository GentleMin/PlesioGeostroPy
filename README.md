# Data Assimilation based on Plesio-Geostrophy (PG) Model


This repo offers a realization for the Plesio-geostrophy (PG) model, a reduced dimensional model for MHD equations in an axisymmetric geometry with nearly geostrophic velocity ansatz. The model is proposed by [Jackson and Maffei, 2020](https://doi.org/10.1098/rspa.2020.0513).

Ingredients of the system:
- Navier-Stokes equation (optionally with uniform viscous diffusivity) in the (rapidly) rotating frame
- Magnetic induction equation (optionally with uniform magnetic diffusivity)
- Plesio-geostrophy Ansatz (perhaps first proposed by Schaeffer and Cardin?)

$$\mathbf{u} = \frac{1}{H}\nabla\times \psi \hat{\mathbf{z}} + \frac{z}{sH^2}\frac{dH}{ds}\frac{\partial \psi}{\partial \phi} \hat{\mathbf{z}}$$

Systems fulfilling these conditions can be exactly described using the PG model.

## Eigenvalue problems

*Ongoing work*

## Data Assimilation

*Future work*

## Current code implementation

Module vector calculus in 3D `pg_utils.sympy_supp`
![uml_v3d](./out/docs/classes_vector_calculus/classes_vector_calculus.svg)

Module PG model `pg_utils.pg_model`
![uml_pg_model](./out/docs/classes_pg_model/classes_pg_model.svg)

