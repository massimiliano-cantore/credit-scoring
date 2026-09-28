"""Valutazione di un richiedente con l'albero decisionale addestrato nel notebook.

`prepara` è identica a quella del notebook; `CreditScorer.valuta` restituisce esito, probabilità e motivazioni.
"""
import json
import os

import numpy as np
import pandas as pd

NUM = ['REDDITO', 'ETA', 'ANZIANITA', 'CNT_CHILDREN']
BIN = ['OCCUPATO', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
CAT = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
FEATURES = NUM + BIN + CAT

ETICHETTE = {'REDDITO': 'Reddito annuo', 'ETA': 'Età', 'ANZIANITA': 'Anzianità lavorativa'}


def prepara(df):
    """Trasforma i dati grezzi (stesso formato del CSV originale) nelle variabili del modello."""
    out = pd.DataFrame(index=df.index)
    out['REDDITO'] = df['AMT_INCOME_TOTAL'].astype(float)
    out['ETA'] = -df['DAYS_BIRTH'] / 365.25
    occupato = df['DAYS_EMPLOYED'] <= 0
    out['OCCUPATO'] = occupato.astype(int)
    out['ANZIANITA'] = np.where(occupato, -df['DAYS_EMPLOYED'] / 365.25, 0.0)
    out['CNT_CHILDREN'] = df['CNT_CHILDREN']
    for c in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        out[c] = (df[c] == 'Y').astype(int)
    for c in ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']:
        out[c] = df[c].astype(int)
    for c in CAT:
        out[c] = df[c].fillna('Non indicato').astype(str).astype(object)
    return out[FEATURES]


def _fmt(col, v):
    return f'{v:,.0f}'.replace(',', '.') if col == 'REDDITO' else f'{v:.1f} anni'


class CreditScorer:
    def __init__(self, model_dir):
        import joblib
        self.model = joblib.load(os.path.join(model_dir, 'credit_tree_pipeline.joblib'))
        self.config = json.load(open(os.path.join(model_dir, 'config.json')))
        self.soglia = self.config['soglia']
        self.requisiti = self.config['soglie_requisiti']

    def valuta(self, cliente_raw):
        """cliente_raw: dizionario con le colonne del CSV originale. Ritorna (esito, probabilità, motivazioni)."""
        x = prepara(pd.DataFrame([cliente_raw]))
        p = float(self.model.predict_proba(x)[0, 1])
        mancanti = [(c, float(x[c].iloc[0]), s) for c, s in self.requisiti.items() if x[c].iloc[0] < s]
        esito = 'IDONEO' if (p >= self.soglia and not mancanti) else 'NON IDONEO'
        motivi = [f'{ETICHETTE[c]}: {_fmt(c, v)}, sotto il requisito minimo di {_fmt(c, s)}' for c, v, s in mancanti]
        if not mancanti and esito == 'NON IDONEO':
            motivi.append('Requisiti minimi rispettati, ma il profilo ricade in un gruppo con bassa affidabilità storica')
        if esito == 'IDONEO':
            motivi.append(f'Tutti i requisiti rispettati: tra i profili simili il {p:.0%} è risultato affidabile. '
                          'Decisione finale da confermare con lo storico creditizio.')
        return esito, p, motivi
