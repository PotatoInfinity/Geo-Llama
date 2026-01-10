# **Phase Transition Analysis & Structural Scaling Laws**

The following document summarizes our latest empirical validation using the **2D Ising Model benchmark**. We demonstrate that embedding physical symmetries within a **$Cl_{4,1}$ Conformal Manifold** allows the architecture to resolve critical phenomena that standard "Flat-AI" (Euclidean) models fail to capture even at $8.9\times$ the parameter scale.
<img width="1790" height="989" alt="unknown" src="https://github.com/user-attachments/assets/6ec9ac16-dc17-4007-84de-d1404db1c298" />
<img width="1789" height="989" alt="unknown" src="https://github.com/user-attachments/assets/d893c1c4-5297-4186-a7c9-ce642a900688" />

---

## **1. Executive Summary**
We evaluated **Geo-Llama** against standard Transformer architectures in the classification of 2D Ising model phase transitions. The objective was to identify three distinct thermodynamic states: **Ordered ($T=1.0$)**, **Disordered ($T=5.0$)**, and the **Critical Point ($T=2.269$)**.

**Key Finding:** Geo-Llama resolves the Critical Point with **95.6% accuracy**, while the Euclidean baseline fails to exceed **50.2%** (stochastic noise) at a comparable parameter scale ($1.2\times$). To match Geo-Llama's "Structural Intelligence" at the critical point, a standard Transformer requires an **$8.9\times$ increase in parameter density**, effectively breaking the traditional Scaling Law.

---

## **2. The "Scaling Law" Disruption**

### **2.1 Parameter Efficiency (Seed 553)**
The following table highlights the capacity gap required for a standard model to match the geometric resolution of Geo-Llama.

| Metric | Geo-Llama | Vanilla (Capacity Match) | Scaling Factor |
| :--- | :--- | :--- | :--- |
| **Parameters** | **18,307** | 162,243 | **~8.9x Larger** |
| **Storage (Disk)** | **71.5 KB** | 633.8 KB | **~8.9x Larger** |
| **Critical Phase Acc.** | **95.6%** | 96.2% (Matched) | Parity at $9\times$ size |

### **2.2 Learning Dynamics: The "Grokking" Threshold**
As shown in the **Few-Shot Accuracy (N=45)** curves, Geo-Llama exhibits a unique "Initialization Lag":
- **Vanilla Model:** Learns surface-level correlations instantly but plateaus at an "Information Floor" below the critical resolution.
- **Geo-Llama:** Remains at baseline for a few epochs before achieving a **Topological Alignment**. Once the $Cl_{4,1}$ rotors align with the lattice's conformal symmetry, accuracy "groks" to near-perfection in fewer than 5 epochs.

---



## **4. Technical Implementation Details**

- **Conformal Null Lifting:** Ising spins are lifted to $n_o$ (origin) and $n_\infty$ (infinity) basis blades.
- **Bivector Attention:** The attention mechanism is biased by the **Wedge Product ($Q \wedge K$)**, allowing the model to attend to the *plane of spin-correlation* rather than just scalar intensity.
- **Manifold Stabilization:** To prevent "Geometric Soup" (numerical drift), we utilize a periodic **Gram-Schmidt Orthonormalization** on the Context Rotor $\Psi$.
---
**Author:** Trương Minh Huy  
**Subject:** Geometric Deep Learning, $Cl_{4,1}$ Manifolds, Ising Criticality
