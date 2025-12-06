import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

class Data_Cleaning:
    def __init__(self,csv_file, sep = ','):
        self.csv_file = csv_file
        self.df = None
        self.sep = sep
        self.cols = None
        self.x_num = None
        self.x_str = None
        self.x = None
        self.y = None
        self.target = 'target'
        self.reply = None

    # Est ce que le df contient de header ou non
    def isHeader(self):
        if self.reply == 'yes':
            self.df = pd.read_csv(self.csv_file, sep = self.sep)
        else:
            self.df = pd.read_csv(self.csv_file, header = None)
            self.cols = [c.strip() for c in self.cols.split(',')]
            self.df.columns = self.cols
    
    # On fait la separation les colonnes numeriques et les colonnes catigoriales
    def separation_xnum_xstr(self):
        self.x_num = self.df.select_dtypes(include = [np.number])
        self.x_str = self.df.select_dtypes(exclude = [np.number])

        # Remplissage des valeurs manquantes par le mode
        for i in self.x_str.columns:
            self.x_str[i] = self.x_str[i].fillna(self.x_str[i].mode()[0])
    # Traiter les colonnes de date
    def extraction_date(self):
        date_cols = []
    
        for col in self.x_str.columns:
    
            # Si la colonne ne contient aucun chiffre, on l’ignore
            if self.x_str[col].astype(str).str.contains(r'\d', regex=True, na=False).sum() == 0:
                continue
    
            # On essaye de convertir les colonnes en datetime
            converted = pd.to_datetime(self.x_str[col], errors='coerce')
    
            # Si au moins une valeur a ete convertie correctement
            if converted.notna().sum() > 0:
                self.x_str[col] = converted
                date_cols.append(col)
    
        for col in date_cols:
            self.x_str[col + '_annee'] = self.x_str[col].dt.year
            self.x_str[col + '_mois'] = self.x_str[col].dt.month
            self.x_str[col + '_jour'] = self.x_str[col].dt.day
            self.x_str[col + '_jrsemaine'] = self.x_str[col].dt.weekday
    
            # Supprimer la colonne originale
            self.x_str.drop(columns = col)
    
    # On fait l'encodeage des colonnes categoriale par LabelEncoder
    def encodage(self):
        encoder = LabelEncoder()

        for i in self.x_str.columns:
            self.x_str[i] = encoder.fit_transform(self.x_str[i])

        # On fait la concatination des colonnes encodées (initialement categoriale) avec les autres colonnes numeriques
        self.x = pd.concat([self.x_num,self.x_str],axis = 1)

    # La suppression de colonne ID
    def suppression_id(self):
        if self.target != '':
            self.x = self.x.drop(columns = self.target)

    # Choisir la methode de remplissage des valeurs manquantes
    def val_manq(self):
        if self.reply == 'intelligente':
            impute = IterativeImputer(random_state = 42)
            self.x = pd.DataFrame(impute.fit_transform(self.x), columns = self.x.columns)
        elif self.reply == 'moyenne':
            self.x = self.x.fillna(self.x.mean())
        else:
            self.x = self.x.fillna(self.x.median())

    # La suppression des observation doublons
    def duplication(self):
        self.x = self.x.drop_duplicates()
        
    # Pour traiter les valeurs aburantes, j'ai utilisé le "capping"
    def outlier(self,col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)

        IQR = Q3 - Q1

        Fb = Q1 - 1.5 * IQR
        Fh = Q3 + 1.5 * IQR

        return col.clip(lower = Fb, upper = Fh)
    
    def remp_outlier(self):
        for col in self.x.columns : 
            self.x[col] = self.outlier(self.x[col])

    # Séparation des caracteristiques (x) et la variable cible (y)
    def separation_x_y(self):
        self.y = self.x[self.target]
        self.x = self.x.drop(columns = self.target)
    

    # Est ce que l'utilisateur souhaite de faire la standarisation ou non
    def standarisation(self):
        scale = StandardScaler()

        if self.reply == 'yes':
            self.x = pd.DataFrame(scale.fit_transform(self.x),columns = self.x.columns)
            
    # Retourner la nouvelle dataframe nettoyée
    def df_final(self):
        self.df = pd.concat([self.x,self.y], axis = 1)
        self.df.reset_index(drop = True)
        return self.df