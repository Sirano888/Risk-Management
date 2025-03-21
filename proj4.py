import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

# Classe pour résoudre les EDP avec Crank-Nicholson
class EDPSolver:
    def __init__(self, aProc, bProc, cProc, dProc, tmin, tmax, Nt, xmin, xmax, Nx, theta, tminBound, xminBound=None, xmaxBound=None, DxmaxBound=None):
        self.aProc = aProc
        self.bProc = bProc
        self.cProc = cProc
        self.dProc = dProc
        self.tmin = tmin
        self.tmax = tmax
        self.Nt = Nt
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.theta = theta
        self.tminBound = tminBound
        self.xminBound = xminBound if xminBound else lambda t, x: 0
        self.xmaxBound = xmaxBound if xmaxBound else lambda t, x: 0
        self.DxmaxBound = DxmaxBound if DxmaxBound else lambda t, x: 0

    def solve(self):
        dt = (self.tmax - self.tmin) / (self.Nt - 1)
        dx = (self.xmax - self.xmin) / (self.Nx - 1)
        x = np.linspace(self.xmin, self.xmax, self.Nx)
        t = np.linspace(self.tmin, self.tmax, self.Nt)
        V = np.zeros((self.Nt, self.Nx))

        # Conditions aux limites initiales
        V[0, :] = [self.tminBound(t[0], xi) for xi in x]

        # Coefficients pour Crank-Nicholson
        for n in range(self.Nt - 1):
            A = np.zeros((self.Nx, self.Nx))
            b = np.zeros(self.Nx)
            for i in range(1, self.Nx - 1):
                a = self.aProc(t[n], x[i])
                b_proc = self.bProc(t[n], x[i])
                c = self.cProc(t[n], x[i])
                alpha = a / (dx ** 2)
                beta = b_proc / (2 * dx)
                gamma = c

                A[i, i-1] = -self.theta * dt * (alpha - beta)
                A[i, i] = 1 + self.theta * dt * (2 * alpha + gamma)
                A[i, i+1] = -self.theta * dt * (alpha + beta)
                b[i] = (1 - (1 - self.theta) * dt * (2 * alpha + gamma)) * V[n, i] + \
                       (1 - self.theta) * dt * (alpha - beta) * V[n, i-1] + \
                       (1 - self.theta) * dt * (alpha + beta) * V[n, i+1] - \
                       dt * self.dProc(t[n], x[i])

            # Conditions aux limites
            A[0, 0] = 1
            b[0] = self.xminBound(t[n+1], x[0])
            A[-1, -1] = 1
            b[-1] = self.xmaxBound(t[n+1], x[-1]) + dx * self.DxmaxBound(t[n+1], x[-1])

            V[n+1, :] = np.linalg.solve(A, b)

        return t, x, V

# Interface Tkinter
class EDPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Projet 4 - Résolution d'EDP (Telecom Paris)")
        self.root.geometry("800x600")

        # Frame pour les paramètres
        param_frame = ttk.LabelFrame(self.root, text="Paramètres", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        self.model_var = tk.StringVar(value="Black & Scholes")
        ttk.Label(param_frame, text="Modèle :").grid(row=0, column=0, padx=5, pady=5)
        ttk.OptionMenu(param_frame, self.model_var, "Black & Scholes", "Black & Scholes", "CIR", "Merton", command=self.update_params).grid(row=0, column=1, padx=5, pady=5)

        self.params = {}
        self.entries = {}
        self.default_params = {
            "Black & Scholes": {"K": 100, "sigma": 0.20, "tau": 0.25, "r": 0.08, "b": -0.04, "theta": 0.5, "xmin": 50, "xmax": 150, "Nx": 201, "Nt": 1000},
            "CIR": {"kappa": 0.8, "theta": 0.10, "sigma": 0.5, "lambda": 0.05, "theta_solver": 0.5, "xmin": 0, "xmax": 1, "Nx": 51, "Nt": 101, "tmax": 5},
            "Merton": {"a": 0.95, "b": 0.10, "sigma": 0.2, "lambda": 0.05, "theta": 0.5, "xmin": -1, "xmax": 1, "Nx": 101, "Nt": 1001, "tmax": 5}
        }

        self.param_frame = ttk.Frame(param_frame)
        self.param_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.update_params("Black & Scholes")  # Initialisation avec Black & Scholes

        # Bouton pour lancer le calcul
        ttk.Button(self.root, text="Lancer le calcul", command=self.run_calculation).pack(pady=10)

        # Frame pour les graphiques
        self.plot_frame = ttk.LabelFrame(self.root, text="Résultats", padding=10)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def update_params(self, model):
        # Supprimer les anciens widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        # Réinitialiser le dictionnaire des entrées
        self.entries = {}
        
        # Charger les paramètres par défaut pour le modèle sélectionné
        params = self.default_params[model]
        self.params = params.copy()
        
        # Créer les nouveaux champs de saisie
        row = 0
        for key, value in params.items():
            ttk.Label(self.param_frame, text=f"{key} :").grid(row=row, column=0, padx=5, pady=2)
            entry = ttk.Entry(self.param_frame)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, padx=5, pady=2)
            self.entries[key] = entry
            row += 1

    def get_params(self):
        params = {}
        for key, entry in self.entries.items():
            try:
                params[key] = float(entry.get())
            except ValueError:
                messagebox.showerror("Erreur", f"Valeur invalide pour {key}")
                return None
        return params

    def run_calculation(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        params = self.get_params()
        if not params:
            return

        model = self.model_var.get()
        if model == "Black & Scholes":
            solver = EDPSolver(
                aProc=lambda t, x: 0.5 * (params["sigma"] ** 2) * (x ** 2),
                bProc=lambda t, x: params["b"] * x,
                cProc=lambda t, x: params["r"],
                dProc=lambda t, x: 0,
                tmin=0, tmax=params["tau"], Nt=int(params["Nt"]),
                xmin=params["xmin"], xmax=params["xmax"], Nx=int(params["Nx"]),
                theta=params["theta"],
                tminBound=lambda t, x: max(x - params["K"], 0),
                xminBound=lambda t, x: 0,
                xmaxBound=lambda t, x: max(x - params["K"], 0),
                DxmaxBound=lambda t, x: 1
            )
        elif model == "CIR":
            mu = lambda t, x: params["kappa"] * (params["theta"] - x)
            sigma = lambda t, x: params["sigma"] * np.sqrt(max(x, 0))
            lambd = lambda t, x: params["lambda"] * np.sqrt(max(x, 0)) / params["sigma"]
            solver = EDPSolver(
                aProc=lambda t, x: 0.5 * (sigma(t, x) ** 2),
                bProc=lambda t, x: mu(t, x) - lambd(t, x) * sigma(t, x),
                cProc=lambda t, x: x,
                dProc=lambda t, x: 0,
                tmin=0, tmax=params["tmax"], Nt=int(params["Nt"]),
                xmin=params["xmin"], xmax=params["xmax"], Nx=int(params["Nx"]),
                theta=params["theta_solver"],
                tminBound=lambda t, x: 1
            )
        elif model == "Merton":
            bprime = params["b"] - params["lambda"] * params["sigma"] / params["a"]
            solver = EDPSolver(
                aProc=lambda t, x: 0.5 * (params["sigma"] ** 2),
                bProc=lambda t, x: params["a"] * (bprime - x),
                cProc=lambda t, x: x,
                dProc=lambda t, x: 0,
                tmin=0, tmax=params["tmax"], Nt=int(params["Nt"]),
                xmin=params["xmin"], xmax=params["xmax"], Nx=int(params["Nx"]),
                theta=params["theta"],
                tminBound=lambda t, x: 1
            )

        t, x, V = solver.solve()

        # Affichage des résultats
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, V[-1, :], label=f"Solution à t={t[-1]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("V(t,x)")
        ax.set_title(f"Résolution EDP - {model}")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EDPApp(root)
    root.mainloop()