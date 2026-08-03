import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import matplotlib.patches as patches


a0 = 0.0529177 # nm
Ha = 27.2114 # eV

def g(x, y, z):
    return 1/(2 * np.pi) * np.arctan(x*y / (z * np.sqrt(x**2 + y**2 + z**2)))

def sym_g(x, y, z):
    return 1/(2 * sp.pi) * sp.atan(x*y / (z * sp.sqrt(x**2 + y**2 + z**2)))

class SquareGate:
    def __init__(self, x0, y0, z0, Lx, Ly):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.Lx = Lx
        self.Ly = Ly

        self.L = self.x0 - self.Lx/2
        self.R = self.x0 + self.Lx/2
        self.B = self.y0 - self.Ly/2
        self.T = self.y0 + self.Ly/2

        self.vi = 0.0

    def add_voltage(self, vi):
        self.vi = vi

    def Vi(self, x, y):
        return self.vi * (
            g(x - self.L, y - self.B, -self.z0)
            + g(x - self.L, self.T - y, -self.z0)
            + g(self.R - x, y - self.B, -self.z0)
            + g(self.R - x, self.T - y, -self.z0)
        )

    def to_au(self):
        self.x0 /= a0
        self.y0 /= a0
        self.z0 /= a0
        self.Lx /= a0
        self.Ly /= a0

        self.L = self.x0 - self.Lx/2
        self.R = self.x0 + self.Lx/2
        self.B = self.y0 - self.Ly/2
        self.T = self.y0 + self.Ly/2

        self.vi /= Ha

    def plot(self, ax=None, **kwargs):
        """Plot the gate as a rectangle."""
        if ax is None:
            ax = plt.gca()
        rect = patches.Rectangle(
            (self.L, self.B),  # bottom-left corner
            self.Lx, self.Ly,  # width, height
            linewidth=2,
            alpha=1,
            **kwargs
        )
        ax.add_patch(rect)
        ax.text(
            self.x0, self.y0, f"V={self.vi:.1f}",
            ha="center", va="center", color="black"
        )
        return ax

class SymSquareGate:
    def __init__(self, x0, y0, z0, Lx, Ly):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.Lx = Lx
        self.Ly = Ly

        self.L = sp.nsimplify(self.x0 - self.Lx/2)
        self.R = sp.nsimplify(self.x0 + self.Lx/2)
        self.B = sp.nsimplify(self.y0 - self.Ly/2)
        self.T = sp.nsimplify(self.y0 + self.Ly/2)

        self.vi = 0.0

    def add_voltage(self, vi):
        self.vi = vi

    def Vi(self, x, y):
        return self.vi * (
            sym_g(x - self.L, y - self.B, -self.z0)
            + sym_g(x - self.L, self.T - y, -self.z0)
            + sym_g(self.R - x, y - self.B, -self.z0)
            + sym_g(self.R - x, self.T - y, -self.z0)
        )


class Gates:
    def __init__(self, gates: list[SquareGate]):
        self.gates = gates
        self.change_to_au = False

    def add_voltages(self, voltages: list[float]):
        for gate, vi in zip(self.gates, voltages):
            gate.add_voltage(vi)

    def Vi(self, x, y):
        return sum(gate.Vi(x, y) for gate in self.gates)
    
    def to_au(self):
        for gate in self.gates:
            gate.to_au()

        self.change_to_au = True

    def plot(self, ax=None, colors=None):
        """Plot all gates on the same figure."""
        axes_none = ax is None
        if axes_none:
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for i, gate in enumerate(self.gates):
            gate.plot(ax=ax, color=colors[i] if colors else None)
            x_min = min(x_min, gate.L)
            x_max = max(x_max, gate.R)
            y_min = min(y_min, gate.B)
            y_max = max(y_max, gate.T)

        x_min = x_min - 0.1 * (x_max - x_min)
        x_max = x_max + 0.1 * (x_max - x_min)
        y_min = y_min - 0.1 * (y_max - y_min)
        y_max = y_max + 0.1 * (y_max - y_min)

        if axes_none:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        ax.set_aspect("equal")
        if not self.change_to_au:
            ax.set_xlabel("$x$ $[\mathrm{nm}]$")
            ax.set_ylabel("$y$ $[\mathrm{nm}]$")
        else:
            ax.set_xlabel("$x$ $[\mathrm{a.u}]$")
            ax.set_ylabel("$y$ $[\mathrm{a.u}]$")
        return ax

    def plot_potential(self, X, Y, ax=None, colors=None):
        if self.change_to_au:
            X /= a0
            Y /= a0

        V = self.Vi(X, Y)
        units_label = 'm'


        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        c = ax.contourf(X, Y, V, levels=50, cmap="inferno")
        ax.set_aspect("equal")
        if not self.change_to_au:
            fig.colorbar(c, ax=ax, label=f"$V(x,y)$ $[\mathrm{{{units_label}eV}}]$")
            ax.set_xlabel("$x$ $[\mathrm{nm}]$")
            ax.set_ylabel("$y$ $[\mathrm{nm}]$")
        else:
            fig.colorbar(c, ax=ax, label=f"$V(x,y)$ $[\mathrm{{{units_label}Ha}}]$")
            ax.set_xlabel("$x$ $[\mathrm{a.u}]$")
            ax.set_ylabel("$y$ $[\mathrm{a.u}]$")
        return ax
    
    def plot_slice(self, x, y, ax=None):
        if self.change_to_au:
            x /= a0
            y /= a0

        V = self.Vi(x, y)
        units_label = 'm'

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        ax.plot(x, V, color='blue')
        if not self.change_to_au:
            ax.set_xlabel("$x$ $[\mathrm{nm}]$")
            ax.set_ylabel(f"$V(x,0)$ $[\mathrm{{{units_label}eV}}]$")
        else:
            ax.set_xlabel("$x$ $[\mathrm{a.u}]$")
            ax.set_ylabel(f"$V(x,0)$ $[\mathrm{{{units_label}Ha}}]$")
        return ax


class SymGates:
    def __init__(self, gates: list[SymSquareGate]):
        self.gates = gates

    def add_voltages(self, voltages: list[float]):
        for gate, vi in zip(self.gates, voltages):
            gate.add_voltage(vi)

    def Vi(self, x, y):
        return sp.Add(*[gate.Vi(x, y) for gate in self.gates])
    
def simplify_terms(expr, f=lambda x:x):
    terms = expr.as_ordered_terms()
    new_terms = []
    for t in terms:
        t = f(t.factor())
        new_terms.append(t)
    return sp.Add(*new_terms)