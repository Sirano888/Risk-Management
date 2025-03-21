import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class OptionPricingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Option Pricing Interface")
        self.geometry("800x700")

        self.option_params = {
            'Vanille': ['S0', 'K', 'T', 'r', 'sigma', 'n_steps', 'n_simulations'],
            'Tunnel': ['S0', 'K', 'L', 'H', 'T', 'r', 'sigma', 'n_steps', 'n_simulations'],
            'Himalaya': ['S0', 'T', 'r', 'sigma', 'n_assets', 'n_steps', 'n_simulations'],
            'Napoléon': ['S0', 'T', 'r', 'sigma', 'n_steps', 'barrier', 'n_simulations']
        }

        self.create_widgets()

    def create_widgets(self):
        
        label = tk.Label(self, text="Choisissez le type d'option :", font=("Arial", 12))
        label.pack(pady=5)

        self.option_type = ttk.Combobox(self, values=list(self.option_params.keys()))
        self.option_type.pack(pady=5)
        self.option_type.bind("<<ComboboxSelected>>", self.show_fields)

        self.params_frame = tk.Frame(self)
        self.params_frame.pack(pady=10)

        self.calc_button = tk.Button(self, text="Calculer", command=self.calculate)
        self.calc_button.pack(pady=10)

        self.result_label = tk.Label(self, text="", font=("Arial", 12))
        self.result_label.pack(pady=5)

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack()

        
    def show_fields(self, event):
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        option_type = self.option_type.get()
        params = self.option_params.get(option_type, [])

        self.entries = {}

        for param in params:
            row = tk.Frame(self.params_frame)
            row.pack(side="top", fill="x", pady=2)

            label = tk.Label(row, text=param + ":", width=15, anchor="w")
            label.pack(side="left")

            entry = tk.Entry(row)
            entry.pack(side="right", expand=True, fill="x")

            self.entries[param] = entry
            
    def calculate(self):
        option_type = self.option_type.get()
        if not option_type:
            self.result_label.config(text="Veuillez sélectionner un type d'option.")
            return

        try:
            params = {param: float(entry.get()) for param, entry in self.entries.items()}

            if 'n_steps' in params:
                params['n_steps'] = int(params['n_steps'])
            if 'n_simulations' in params:
                params['n_simulations'] = int(params['n_simulations'])
            if 'n_assets' in params:
                params['n_assets'] = int(params['n_assets'])
            if 'barrier' in params:
                params['barrier'] = float(params['barrier'])

            if option_type == 'Vanille':
                price, error = self.monte_carlo_option_vanilla(**params)
            elif option_type == 'Tunnel':
                price, error = self.monte_carlo_option_tunnel(**params)
            elif option_type == 'Himalaya':
                price, error = self.monte_carlo_option_himalaya(**params)
            elif option_type == 'Napoléon':
                price, error = self.monte_carlo_option_napoleon(**params)
            else:
                self.result_label.config(text="Option inconnue.")
                return

            self.result_label.config(
                text=f"Prix de l'option : {price:.4f}\nErreur à 99% : {error:.4f}"
            )

            self.plot_convergence(option_type, params)

        except Exception as e:
            self.result_label.config(text=f"Erreur : {e}")            
            
    def monte_carlo_option_vanilla(self, S0, K, T, r, sigma, n_steps, n_simulations):
        """
        Pricing d'une option vanille de type Call avec la méthode de Monte Carlo.
        On va discrétiser l'intervalle [0,T] en n_steps pas de temps.
        On va donc vectoriser pour !!! avec numpy!!!! attention, grace a la vectorisation on gagne énormement
        en temps de calcul mais on perd aussi en précision (numpy encode les 
        flottants sur moins de bits) !! mais vu que cest pour le cours et pas 
        dans un cadre professionel ce n'est pas grave.
        """
        #np.random.seed(1) # Pour reproductibilité
        dt = T / n_steps # Pas de temps
        dW = np.random.randn(n_simulations, n_steps) * np.sqrt(dt) # matrice 2D des incréments browniens: axe 0: simulations, axe 1: pas de temps
        S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * dW, axis=1)) # matrice 2D des valeurs des St: axe 0: simulations, axe 1: pas de temps
        ST = S[:, -1]  # Prix à l'échéance, matrice 1D des valeurs de S_T: axe 0; simulation
        payoffs = np.maximum(ST - K, 0)
        option_price = np.exp(-r * T) * np.mean(payoffs) # Prix de l'option (actualisation avec le taux sans risque)
        error_99 = norm.ppf(0.99) * np.std(payoffs) / np.sqrt(n_simulations)     # Erreur à 99% (intervalle de confiance)
        return option_price, error_99

    def monte_carlo_option_tunnel(self, S0, K, L, H, T, r, sigma, n_steps, n_simulations):
        #np.random.seed(1)
        dt = T / n_steps
        dW = np.random.randn(n_simulations, n_steps) * np.sqrt(dt)
        S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * dW, axis=1))
        knock_out = (S < L) | (S > H)     # Vérification si le prix franchit une des barrières à n'importe quel moment
        knocked_out = np.any(knock_out, axis=1) # Vrai si le prix sort du tunnel à un moment donné
        ST = S[:, -1] # Prix à l'échéance
        payoffs = np.where(~knocked_out, np.maximum(ST - K, 0), 0)
        option_price = np.exp(-r * T) * np.mean(payoffs)
        error_99 = norm.ppf(0.99) * np.std(payoffs) / np.sqrt(n_simulations)
        return option_price, error_99

    def monte_carlo_option_himalaya(self, S0, T, r, sigma, n_assets, n_steps, n_simulations):
        #np.random.seed(1)
        dt = T / n_steps
        dW = np.random.randn(n_simulations, n_steps, n_assets) * np.sqrt(dt) # matrice 3D des incréments browniens: axe0: sim, axe1:pas de temps, axe 2:asset considéré
        S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * dW, axis=1)) # matrice 3D des St: axe0: sim, axe1:pas de temps, axe 2:asset considéré
        payoffs = np.zeros(n_simulations)
        for i in range(n_steps):
            # Vérification si tous les actifs sont déjà retirés
            if np.all(S[:, i, :] == -np.inf): #matrice 2D: axe 0 sim, axe 1: assets considéré
                break
            best_asset = np.argmax(S[:, i, :], axis=1)
            if i > 0:
                prev_price = S[np.arange(n_simulations), i - 1, best_asset]
                curr_price = S[np.arange(n_simulations), i, best_asset]
                performance = np.maximum(curr_price / prev_price - 1, 0) #on retient le max entre curr_price/prev_price -1 et 0
                payoffs += performance
            S[np.arange(n_simulations), :, best_asset] = -np.inf
        option_price = np.exp(-r * T) * np.mean(payoffs)
        error_99 = norm.ppf(0.99) * np.std(payoffs) / np.sqrt(n_simulations)
        return option_price, error_99

    def monte_carlo_option_napoleon(self, S0, T, r, sigma, n_steps, barrier, n_simulations):
        #np.random.seed(1)
        dt = T / n_steps
        dW = np.random.randn(n_simulations, n_steps) * np.sqrt(dt)
        S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * dW, axis=1))
        knocked_out = np.any(S < barrier, axis=1)
        avg_price = np.mean(S, axis=1)
        payoffs = np.where(knocked_out, 0, np.maximum(avg_price - S0, 0))
        option_price = np.exp(-r * T) * np.mean(payoffs)
        error_99 = norm.ppf(0.99) * np.std(payoffs) / np.sqrt(n_simulations)
        return option_price, error_99
    
    def plot_convergence(self, option_type, params):
        # Nettoyage du graphique précédent
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        prices = []
        n_sim_list = np.arange(100, params['n_simulations'] + 100, 100)

        for n in n_sim_list:
            if option_type == 'Vanille':
                price, _ = self.monte_carlo_option_vanilla(
                    params['S0'], params['K'], params['T'], params['r'], 
                    params['sigma'], params['n_steps'], n)
            elif option_type == 'Tunnel':
                price, _ = self.monte_carlo_option_tunnel(
                    params['S0'], params['K'], params['L'], params['H'], params['T'], 
                    params['r'], params['sigma'], params['n_steps'], n)
            elif option_type == 'Himalaya':
                price, _ = self.monte_carlo_option_himalaya(
                    params['S0'], params['T'], params['r'], params['sigma'], 
                    params['n_assets'], params['n_steps'], n)
            elif option_type == 'Napoléon':
                price, _ = self.monte_carlo_option_napoleon(
                    params['S0'], params['T'], params['r'], params['sigma'], 
                    params['n_steps'], params['barrier'], n)
            prices.append(price)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(n_sim_list, prices, label='Prix estimé par Monte Carlo')
        ax.axhline(y=prices[-1], color='r', linestyle='--', label='Valeur finale')
        ax.set_title(f'Convergence du prix de l\'option {option_type}')
        ax.set_xlabel('Nombre de simulations')
        ax.set_ylabel('Prix estimé')
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
if __name__ == "__main__":
    app = OptionPricingApp()
    app.mainloop()
