# Physics-Informed Neural Network for a Mass-Spring-Damper System

## Project Overview

This project uses a **Physics-Informed Neural Network (PINN)** to learn the motion of a mass-spring-damper system governed by a second-order ordinary differential equation (ODE).
A PyTorch implementation of a Physics-Informed Neural Network that solves the free vibration of a single-degree-of-freedom mass-spring-damper system. 
The PINN prediction is compared against a numerical ground truth computed with SciPy's odeint, and several experiments explore how damping, stiffness, loss weights, and boundary-condition handling affect training.

## Problem Statement

The system is governed by the second-order ODE:

m·x''(t) + c·x'(t) + k·x(t) = F(t)
Symbol	Meaning	Value
m	Mass	1.0
c	Damping coefficient	varied (default 0.2)
k	Spring stiffness	varied (default 1.0)
F(t)	External force	0 (free vibration)
x(0), x'(0)	Initial position, velocity	1.0, 0.0
Time domain: t ∈ [0, 15].

Instead of relying only on data, the neural network is trained using:
* The governing physical equation
* Initial/boundary conditions
* Numerically generated solution data

The PINN predictions are then compared with the numerical ground-truth solution to evaluate how well the network learns the physical system.

## Problem Formulation

The mass-spring-damper system is described by:

$$
m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F(t)
$$

where:

* \(m\) = mass
* \(c\) = damping coefficient
* \(k\) = stiffness coefficient
* \(x(t)\) = displacement
* \(F(t)\) = external force

For this project:

```text
m = 1.0
x(0) = 1.0
v(0) = 0.0
F(t) = 0
```

The system therefore represents **free vibration** of a damped mass-spring system.

The second-order ODE is converted into a first-order system for obtaining the numerical solution using `scipy.integrate.odeint`.

---

## Methodology

The project follows these main steps:

1. Define the mass-spring-damper governing equation.
2. Generate a numerical solution using `odeint`.
3. Build a fully connected neural network using PyTorch.
4. Generate collocation points for enforcing the physics.
5. Use automatic differentiation to calculate:

   * \(dx/dt\)
   * \(d^2x/dt^2\)
6. Construct the physics residual.
7. Include initial-condition and supervised data losses.
8. Combine the losses into a total training loss.
9. Train the PINN using the Adam optimizer.
10. Compare PINN predictions with the numerical solution.
11. Investigate the effects of damping, stiffness, loss weights, and boundary-condition treatment.

---

## Neural Network Architecture

The PINN is a fully connected neural network with:

```text
Input:        1 neuron (time t)
Hidden layer: 64 neurons + Tanh
Hidden layer: 64 neurons + Tanh
Hidden layer: 64 neurons + Tanh
Output:       1 neuron (displacement x)
```

The network takes **time \(t\)** as input and predicts the displacement \(x(t)\).

---

## Physics-Informed Loss

The governing equation is incorporated into the training process through the physics residual:

$$
R(t) = m x_{tt} + c x_t + kx - F(t)
$$

The physics loss is:

$$
L_{physics} = \frac{1}{N}\sum_i R(t_i)^2
$$

This encourages the neural network to produce predictions that satisfy the governing differential equation.

---

## Boundary Conditions

Two approaches are implemented.

### Soft Boundary Condition

The network directly predicts \(x(t)\), while the initial conditions are enforced through an additional loss term.

### Hard Boundary Condition

The initial conditions are built directly into the neural-network output:

$$
x(t) = x_0 + v_0t + t^2N(t)
$$

where \(N(t)\) is the neural network output.

This formulation ensures that the initial position and velocity are satisfied by construction.

---

## Total Loss

The training objective combines three components:

$$
L =
\lambda_p L_{physics}
+
\lambda_{bc} L_{BC}
+
\lambda_d L_{data}
$$

where:

* \(L_{physics}\) = physics residual loss
* \(L_{BC}\) = initial-condition loss
* \(L_{data}\) = supervised data loss
* \(\lambda_p\), \(\lambda_{bc}\), \(\lambda_d\) = loss weights

The default values used are:

```text
λp  = 1
λbc = 10
λd  = 1
```

---

## Experiments / Generated Plots
- Solution vs. True – PINN prediction against the numerical solution (c=0.2, k=1.0).
- Effect of Damping – c ∈ {0.1, 0.5, 1.0}.
- Effect of Stiffness – k ∈ {0.5, 1.0, 2.0}.
- Loss Weight Tuning – λ_bc ∈ {1, 10, 100}.
- Loss vs. Epochs – physics, BC, and total loss curves (log scale).
- Hard vs. Soft BC – comparison of both strategies against the truth.
- Absolute Error – |x_true − x_pred| over time.

## Libraries Used

* Python
* PyTorch
* NumPy
* Matplotlib
* SciPy
* `scipy.integrate.odeint`
* Automatic differentiation

---

## Key Concepts Demonstrated

* Physics-Informed Neural Networks (PINNs)
* Neural-network-based ODE solving
* Automatic differentiation
* Physics-based loss functions
* Initial/boundary conditions
* Hard and soft constraint enforcement
* Loss-weight tuning
* Numerical ODE solutions
* Model prediction error analysis
* Mass-spring-damper dynamics

---

## Project Structure

```text
PINN-Mass-Spring-Damper/
│
├── PINN_Mass_Spring_Damper.ipynb
└── README.md
```

---
## Known Limitations / Notes
1. Training is done with full-batch Adam only; adding an L-BFGS refinement stage often improves accuracy.
2. The final Absolute Error plot uses the x_pred variable left over from the most recent loop (the loss-weight experiment). To plot the error for the baseline model, recompute x_pred from the baseline model before that step.
3. Models are retrained from scratch for every experiment and are not seeded; results will vary slightly between runs. Set torch.manual_seed(...) for reproducibility.
4. Accuracy over long time horizons (large t, low damping) can degrade; more collocation points, more epochs, or time-domain splitting may help.

## Conclusion
This project demonstrates how a neural network can learn the solution of a physical dynamical system while being guided by its governing differential equation.
The PINN is evaluated against a numerical ODE solution, and additional experiments investigate how physical parameters, loss weights, and boundary-condition strategies influence the learned solution.
The project provides a practical introduction to combining **deep learning with differential equations and physical constraints**.
