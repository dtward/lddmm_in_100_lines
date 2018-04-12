# lddmm_in_100_lines

This code performs 2D image registration using LDDMM.

It is intended to be used as a jumping off point for building more specialized algorithms.

It parameterizes a diffeomorphism through a time varying velocity vector field.

It uses sum of square error as an error metric.

It uses identitity minus laplacian as a highpass operator for regularization.

The regularization is the norm squared of this operator applied to the velocity field

It assumes isotropic pixels of size "1".


