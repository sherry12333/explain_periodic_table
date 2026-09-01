
# Explain Periodic Table

## 中文

本项目利用**密度泛函理论（Density Functional Theory, DFT）**进行原子电子结构的数值计算，以研究元素周期表中不同元素的能量与电子结构规律。

在计算过程中，使用 **B-spline 基函数**表示电子波函数，并通过**高斯求积（Gaussian Quadrature）**对相关积分进行高精度数值计算。通过自洽迭代求解电子结构，获得不同原子的电子密度、轨道能级以及基态总能量。

进一步地，通过分别计算中性原子与相应离子的总能量，可以得到元素的**电离能（ionization energy）**以及相关的原子结合能特征。比较不同元素的计算结果，可以观察电子壳层和亚壳层逐步填充所产生的周期性变化，例如闭壳层结构具有较高稳定性，而新电子壳层的开始填充会导致能量性质发生明显变化。

通过这些数值结果，本项目从量子力学和电子结构计算的角度解释元素周期表中元素排列背后的物理规律，并验证元素周期性与电子壳层结构之间的关系。

该项目主要涉及：

* Density Functional Theory (DFT)
* Self-consistent electronic-structure calculations
* B-spline basis representation
* Gaussian quadrature
* Atomic binding and total-energy calculations
* Ionization-energy calculations
* Electron-density calculations
* Orbital occupation and electronic-shell structure
* Numerical convergence and accuracy analysis

## English

This project uses **Density Functional Theory (DFT)** to perform numerical atomic electronic-structure calculations and investigate the physical principles underlying the periodic table.

In the numerical implementation, **B-spline basis functions** are used to represent electronic wave functions, while **Gaussian quadrature** is employed to accurately evaluate the required integrals. Self-consistent calculations are performed to obtain the electron density, orbital energies, and ground-state total energies of different atoms.

By calculating and comparing the total energies of neutral atoms and their corresponding ions, quantities such as **ionization energies** and atomic binding-energy characteristics can be obtained. Comparing these properties across different elements reveals systematic variations associated with the filling of electronic shells and subshells. For example, closed-shell configurations generally exhibit enhanced stability, while the beginning of a new electronic shell produces characteristic changes in atomic energy properties.

The project therefore provides a computational and quantum-mechanical explanation of the structure of the periodic table and demonstrates how periodic trends emerge from electronic-shell configurations and atomic electronic structure.

The main topics covered in this project include:

* Density Functional Theory (DFT)
* Self-consistent electronic-structure calculations
* B-spline basis representation
* Gaussian quadrature
* Atomic binding and total-energy calculations
* Ionization-energy calculations
* Electron-density calculations
* Orbital occupation and electronic-shell structure
* Numerical convergence and accuracy analysis
