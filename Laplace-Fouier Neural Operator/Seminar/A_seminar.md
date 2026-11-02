---
title: "For A"
subtitle: "A Laplace–Fourier neural operator with wavenumber-dependent poles"
format: 
  revealjs:
   theme: default
   embed-resources: true
   slide-number: true
---

## 

1. A linear PDE is many ODEs, one per wavenumber. The pole of each ODE is the dispersion relation.
2. We make the poles of LNO depend on the wavenumber: $\mu_n \to \mu_n(k)$.
3. Two test cases: time delay in process dynamics, and Cahn–Hilliard phase separation.

---

## Spatial Fourier Transform

Consider the linear diffusion equation with constant diffusivity $D$:

$$
\frac{\partial u(x,t)}{\partial t}
=
D\frac{\partial^2u(x,t)}{\partial x^2}.
$$

---

Define the Fourier transform with respect to the spatial variable $x$:

$$
\widehat{u}(k,t)
=
\int_{-\infty}^{\infty}
u(x,t)e^{-ikx}\,dx.
$$

---

Assuming sufficient smoothness and vanishing boundary terms,

$$
\mathcal{F}_x\left[
\frac{\partial u}{\partial t}
\right]
=
\frac{\partial\widehat{u}}{\partial t},
\qquad
\mathcal{F}_x\left[
\frac{\partial^2u}{\partial x^2}
\right]
=
-k^2\widehat{u}.
$$

---

The transformed equation is therefore

$$
\frac{d\widehat{u}_k(t)}{dt}
=
-Dk^2\widehat{u}_k(t).
$$

For each fixed wavenumber $k$, the PDE becomes a first-order ODE in time.

---

The solution is

$$
\widehat{u}_k(t) = e^{-Dk^2 t}\,\widehat{u}_k(0).
$$

Each mode decays at its own rate. The rate depends on $k$:

$$
\lambda(k) = -Dk^2 .
$$

This is the **dispersion relation**.

---

## Laplace transform in time

$$
s\,\widehat{U}_k(s) - \widehat{u}_k(0) = \lambda(k)\,\widehat{U}_k(s)
\quad\Longrightarrow\quad
\widehat{U}_k(s) = \frac{\widehat{u}_k(0)}{s - \lambda(k)} .
$$

One pole, at $s = \lambda(k)$.

> For a linear PDE, **the pole of each mode is the dispersion relation.**


---

## What LNO does

LNO writes the kernel as a rational function with **constant** poles:

$$
K_\theta(s) = \sum_{n=1}^{N} \frac{\beta_n}{s - \mu_n} .
$$

In 2D or 3D, the kernel is a **product of per-axis poles**.

But diffusion needs the time pole to change with $k$: $\lambda(k) = -Dk^2$.

A product of fixed poles cannot do that. LNO cannot represent the simplest PDE's dispersion relation.

Recent hybrids (ALNO, LFNO) split transient and steady parts in time, but still keep $\mu_n$ constant.

---

## The idea

$$
\mu_n \;\longrightarrow\; \mu_n(k), \qquad \beta_n \;\longrightarrow\; \beta_n(k)
$$

$$
K_\theta(k, s) = \sum_{n=1}^{N} \frac{\beta_n(k)}{s - \mu_n(k)}
$$

Space by Fourier, time by Laplace. The pole function $\mu_n(k)$ ties them together.

One pole function drives both the transient part $\sum_n \gamma_n(k)e^{\mu_n(k)t}$ and the steady part $K_\theta(k, i\omega)$.

---

## How to write $\mu_n(k)$

**Polynomial** — strong prior, readable directly:
$$
\mu_n(k) = \sum_j c_{n,j}|k|^{2j} + i\,(d_{n,1}k + \dots)
$$

**MLP** — flexible, read by fitting afterwards:
$$
\mu_n(k) = g_\theta(k)
$$

Because $\mu_n(k)$ is continuous in $k$, we can evaluate it at modes never seen in training. Train at $64^2$, test at $256^2$.

---

## Application : 

time delay

Cahn–Hilliard

---

## Peoples on reference

## Overview

| Group | Papers |
|---|---|
| Caltech / NVIDIA | FNO [1], neuraloperator [10] |
| Brown (CRUNCH) | LNO [2], PILNO [3], SSM [6] |
| Korea University | LFNO [4], [5] |
| CMU | S4 [7] |
| Classical | Cahn–Hilliard [8], vector fitting [9] |

# [1] FNO

## [1] FNO — Li et al., ICLR 2021

Each Fourier mode is multiplied by a learned weight $R(k)$.

- Resolution-invariant
- Assumes periodic input
- No structure inside the kernel

[arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

## [1] Author and lab

**Zongyi Li** — PhD, Caltech, 2025 (advisor: Anima Anandkumar)

[Anandkumar group, Caltech](https://tensorlab.cms.caltech.edu/users/anima/)

[neuraloperator library](https://neuraloperator.github.io/dev/) — reference [10]

# [2] LNO

## [2] LNO — Cao, Goswami, Karniadakis (2023)

Kernel is a rational function in Laplace space:

$$K_\theta(s) = \sum_n \frac{\beta_n}{s-\mu_n}$$

- Output = steady part + transient part
- Works with non-periodic input
- In 2D/3D: product of per-axis poles

[arXiv:2303.10528](https://arxiv.org/abs/2303.10528) · Nature Machine Intelligence 2024

## [2] Author and lab

**Qianying Cao** — Dalian University of Technology / Brown Applied Math

[CRUNCH group, Brown](https://sites.brown.edu/crunch-group/)

# [3] PILNO / ALNO

## [3] PILNO — Kim et al., 2026

**ALNO**: keep the pole–residue transient part, replace the steady part with an FNO-style Fourier multiplier.

**PILNO**: add PDE / BC / IC residual losses on top.

Better in small-data and out-of-distribution settings.

[arXiv:2602.12706](https://arxiv.org/abs/2602.12706)

## [3] Author and lab

**Heechang Kim** — POSTECH Mathematics (advisor: Minseok Choi)

Joint work with KAIST and Brown.

[Minseok Choi group, POSTECH](https://aiml-k.github.io)

# [4], [5] LFNO

## [4], [5] LFNO — Ha, Yoon, Lee (2025–2026)

Laplace branch for the transient part, Fourier integral operator for the steady part.

- Strong on ODEs
- On PDEs, about equal to FNO

[4] [NeurIPS 2025 ML4PS workshop version](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_281.pdf)  
[5] [arXiv:2606.07601](https://arxiv.org/abs/2606.07601), full paper

## [4], [5] Author and lab

**Jeongun Ha** — PhD student, Korea University Mathematics

[A.I. + Math Lab (Donghun Lee), Korea University](https://dhl.korea.ac.kr/lab)

# [6] SSM operators

## [6] State-space models as operators — Hu et al., 2024

Uses Mamba as the neural operator. Linear cost in sequence length.

Why we cite it: a rational kernel is close to a diagonal SSM. Reviewers will ask how we differ.

[arXiv:2409.03231](https://arxiv.org/abs/2409.03231)

## [6] Author and lab

**Zheyuan Hu** — NUS Computer Science (advisor: Kenji Kawaguchi)

Joint work with Karniadakis.

[Kawaguchi Deep Learning Lab, NUS](https://ml.comp.nus.edu.sg)

# [7] S4

## [7] S4 — Gu, Goel, Ré, ICLR 2022

Structured state-space model for long sequences. The ancestor of Mamba.

[arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

## [7] Author and lab

**Albert Gu** — Assistant Professor, CMU Machine Learning; Chief Scientist, Cartesia AI

Goomba Lab, CMU

# [8], [9] Classical

## [8] Cahn & Hilliard, 1958

Free energy of a non-uniform system.

Where the gradient term $\kappa|\nabla\phi|^2$ comes from.

[DOI: 10.1063/1.1744102](https://doi.org/10.1063/1.1744102)

## [9] Vector fitting — Gustavsen & Semlyen, 1999

Fit a pole–residue rational function to frequency-response data.

A non-learning version of what LNO does. Our classical baseline for the process problems.

**Bjørn Gustavsen** — SINTEF Energy Research, Norway

[Paper (PDF)](https://www.sintef.no/globalassets/project/vectfit/vf_paper.pdf)
