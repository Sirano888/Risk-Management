import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from scipy.stats import norm
import numpy as np

# Charger le fichier Excel
file_path = 'Credit_Portfolio_New.xlsx'
portfolio = pd.read_excel(file_path, sheet_name='Portfolio')
params = pd.read_excel(file_path, sheet_name='Params')


# Extraire la table des probabilités de défaut (cellules A5:D24 dans la sheet 'Params')
pd_table = params.iloc[4:24, 0:4]
corr_table = params.iloc[51:58, 0:3]

pd_table.columns = ['PD', 'Y1', 'Y3', 'Y5']
corr_table.columns = ['ID','Sector_Name','Correlation']

class RaRoCApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Projet 1 - Outil RaRoC")
        self.geometry("980x600")

        self.create_widgets()

    def create_widgets(self):
        ##########################################
        ### Partie gauche — Tableau scrollable
        table_frame = ttk.Frame(self)
        table_frame.place(x=10, y=10, width=420, height=550)

        columns = ['Id', 'Exposure', 'Rating', 'LGD', 'Sector']
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

        self.table.heading('Id', text='Id')
        self.table.column('Id', width=50, anchor="center")
        self.table.heading('Exposure', text='Exposure')
        self.table.column('Exposure', width=110, anchor="center")
        self.table.heading('Rating', text='Rating')
        self.table.column('Rating', width=50, anchor="center")
        self.table.heading('LGD', text='LGD')
        self.table.column('LGD', width=50, anchor="center")
        self.table.heading('Sector', text='Sector')
        self.table.column('Sector', width=150, anchor="center")


        self.table.pack(side="left", fill="both")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scrollbar.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scrollbar.set)

        # Remplir le tableau
        self.load_portfolio_data()
        ##########################################
        ##########################################






        
        ##########################################
        ############ Partie à droite — Haut séléction des params
        # Partie supérieure gauche — Sélection de l'ID
        input_frame = ttk.LabelFrame(self, text="Sélection du crédit et détails")
        input_frame.place(x=440, y=10, width=510, height=200)
        
        input_frame.grid_columnconfigure(1, weight=1)  

        ttk.Label(input_frame, text="ID du crédit :").grid(row=0, column=0, padx=5, pady=5, sticky = "w")
        self.id_entry = ttk.Entry(input_frame)
        self.id_entry.grid(row=0, column=1, padx=5, pady=5, sticky = "w")
        
        ttk.Label(input_frame, text="Maturité :").grid(row=1, column=0, padx=5, pady=5, sticky = "w")
        self.maturity_choice = ttk.Combobox(input_frame, values=["1 an", "3 ans", "5 ans"], state="readonly")
        self.maturity_choice.set("1 an")
        self.maturity_choice.grid(row=1, column=1, padx=5, pady=5, sticky = "w")
        
        ttk.Label(input_frame, text="Marge d'intéret (%) :").grid(row=2, column=0, padx=5, pady=5, sticky = "w")
        self.marge_interet = ttk.Entry(input_frame)
        self.marge_interet.grid(row=2, column=1, padx=5, pady=5, sticky = "w")
                
        ttk.Label(input_frame, text="Coûts :").grid(row=4, column=0, padx=5, pady=5, sticky = "w")
        self.couts = ttk.Entry(input_frame)
        self.couts.grid(row=4, column=1, padx=5, pady=5, sticky = "w")
        
        ttk.Label(input_frame, text="Seuil de confiance (%) :").grid(row=0, column=2, padx=5, pady=5, sticky = "w")
        self.conf = ttk.Entry(input_frame)
        self.conf.grid(row=0, column=3, padx=5, pady=5, sticky = "w")

        ttk.Label(input_frame, text="Diversification f (1/√ρ par déf) :").grid(row=1, column=2, padx=5, pady=5, sticky = "w")
        self.f = ttk.Entry(input_frame)
        self.f.grid(row=1, column=3, padx=5, pady=5, sticky = "w")

        ttk.Label(input_frame, text="TSR (0 par défaut):").grid(row=2, column=2, padx=5, pady=5, sticky = "w")
        self.tsr = ttk.Entry(input_frame)
        self.tsr.grid(row=2, column=3, padx=5, pady=5, sticky = "w")

        ##########################################
        ##########################################




        # Partie droite milieu — Bouton de calcul
        calc_button = ttk.Button(self, text="CALCULER", command=self.calculate_raroc)
        calc_button.place(x=640, y=240, width=120, height=40)


        ##########################################
        ############ Partie à droite — Bas Résultats détaillés
        self.result_frame = ttk.LabelFrame(self, text="Résultats crédit")
        self.result_frame.place(x=440, y=300, width=510, height=250)

        self.result_labels = {}
        fields1 = ["ID", "Exposure", "Rating", "LGD", "Sector", "PD", "EL"]
        for i, field in enumerate(fields1):
            ttk.Label(self.result_frame, text=f"{field} :").grid(row=i, column=0, padx=5, pady=5, sticky="w")
            label = ttk.Label(self.result_frame, text="—")
            label.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.result_labels[field] = label
            
        fields = [ "PNB", "Corrélation", "f (div. portefeuille)","UL","RaRoC"]
        for i, field in enumerate(fields):
            ttk.Label(self.result_frame, text=f"{field} :").grid(row=i, column=3, padx=5, pady=5, sticky="w")
            label = ttk.Label(self.result_frame, text="—")
            label.grid(row=i, column=4, padx=5, pady=5, sticky="w")
            self.result_labels[field] = label

        ##########################################
        ##########################################


        
        
        
    def load_portfolio_data(self):
        # Affiche les 20 premières lignes du portefeuille
        for _, row in portfolio.iterrows():
            self.table.insert('', 'end', values=(row['Id'], row['Exposure'], row['Rating'], row['LGD'], row['Sector']))
        
        
        
        
        
        
        
        
        
        
    def get_pd_from_rating(self, rating, maturity):
        """
        Fonction pour récupérer la probabilité de défaut (PD) en fonction du rating et de la maturité.
        """
        try:
            # Trouver le PD correspondant au rating
            pd_row = pd_table.loc[pd_table['PD'] == rating]

            if not pd_row.empty:
                if maturity == "1 an":
                    return float(pd_row['Y1'].values[0])
                elif maturity == "3 ans":
                    return float(pd_row['Y3'].values[0])
                elif maturity == "5 ans":
                    return float(pd_row['Y5'].values[0])
            else:
                raise ValueError(f"Rating {rating} non trouvé dans la table de PD.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la récupération du PD : {e}")
            return None

    def get_corr_from_sector(self, sector):
        """
        Fonction pour récupérer la probabilité de défaut (PD) en fonction du rating et de la maturité.
        """
        try:
            # Trouver le PD correspondant au rating
            corr_row = corr_table.loc[corr_table['Sector_Name'] == sector]
            if not corr_row.empty:
                return float(corr_row['Correlation'].values[0])
            else:
                raise ValueError(f"Correlation à {sector} non trouvée dans la table de Correlation.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la récupération de la corrélation : {e}")
            return None



    def calculate_raroc(self):
        try:
            # Récupérer l'ID sélectionné
            selected_id = int(self.id_entry.get())
            selected_credit = portfolio.loc[portfolio['Id'] == selected_id]

            if not selected_credit.empty:
                credit_data = selected_credit.iloc[0]
                
                exposure = credit_data['Exposure']
                lgd = credit_data['LGD']
                rating = credit_data['Rating']
                sector = credit_data['Sector']

                # Récupérer la maturité choisie dans le menu déroulant
                maturity = self.maturity_choice.get()

                # Aller chercher la probabilité de défaut (PD) à partir de la table des ratings
                pd = self.get_pd_from_rating(rating, maturity)
                
                # CALCUL EL
                expected_loss = exposure * lgd * pd
                
                # CALCUL PNB
                nim = float(self.marge_interet.get()) 
                pnb = float(nim)*0.01 * float(exposure) 
                    
                #CALCUL Correlation
                correlation = self.get_corr_from_sector(sector)
                
                #CALCUL UL
                if not self.f.get():
                    f = 1/(np.sqrt(correlation))
                else:
                    f = self.f.get()
                seuil_conf = float(self.conf.get()) * 0.01
                beta =norm.ppf(seuil_conf)
                ul = exposure * lgd * f * beta * np.sqrt(pd * (1 - pd) * correlation) 

                #DEFAUT TSR
                if not self.tsr.get():
                    tsr = 0
                else:
                    tsr = float(self.tsr.get())
                
                #RARC
                raroc = ((pnb - float(self.couts.get()) - expected_loss) / ul) + tsr

                # Mise à jour des résultats dans l'interface
                self.result_labels["ID"].config(text=f"{selected_id}")
                self.result_labels["Exposure"].config(text=f"{exposure:.2f} €")
                self.result_labels["Rating"].config(text=f"{rating}")
                self.result_labels["LGD"].config(text=f"{lgd:.2%}")
                self.result_labels["Sector"].config(text=f"{sector}")
                self.result_labels["PD"].config(text=f"{pd:.6f}")
                self.result_labels["EL"].config(text=f"{expected_loss:.2f} €")
                self.result_labels["PNB"].config(text=f"{pnb}")
                self.result_labels["Corrélation"].config(text=f"{correlation}")
                self.result_labels["f (div. portefeuille)"].config(text=f"{f}")
                self.result_labels["UL"].config(text=f"{ul}")
                self.result_labels["RaRoC"].config(text=f"{raroc}")


            else:
                messagebox.showerror("Erreur", f"Aucun crédit trouvé avec l'ID {selected_id}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du calcul : {e}")
            
            
            
            
            
            
            
            
if __name__ == "__main__":
    app = RaRoCApp()
    app.mainloop()