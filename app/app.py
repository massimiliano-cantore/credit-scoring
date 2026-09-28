"""Demo Gradio: valutazione di una richiesta di carta di credito con motivazioni (Hugging Face Spaces)."""
try:  # hardware ZeroGPU di Hugging Face: richiede almeno una funzione @spaces.GPU
    import spaces
    zero_gpu = spaces.GPU
except ImportError:  # esecuzione locale o su CPU
    def zero_gpu(fn):
        return fn
import os
import sys

import gradio as gr
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from credit_scoring.scoring import CreditScorer
except ImportError:  # su Spaces scoring.py è copiato accanto ad app.py
    from scoring import CreditScorer

MODEL_REPO = os.getenv("MODEL_REPO", "MassimilianoCantore/credit-scoring-tree")
scorer = CreditScorer(os.getenv("MODEL_DIR") or snapshot_download(MODEL_REPO))
cat = scorer.config["categorie"]

REDDITO_TIPO = [("Dipendente", "Working"), ("Lavoratore autonomo / commerciale", "Commercial associate"),
                ("Dipendente pubblico", "State servant"), ("Pensionato", "Pensioner"), ("Studente", "Student")]
ISTRUZIONE = [("Laurea", "Higher education"), ("Laurea non completata", "Incomplete higher"),
              ("Diploma superiore", "Secondary / secondary special"), ("Licenza media", "Lower secondary"),
              ("Titolo accademico (dottorato)", "Academic degree")]
STATO_CIVILE = [("Sposato/a", "Married"), ("Convivente / unione civile", "Civil marriage"), ("Single", "Single / not married"),
                ("Separato/a", "Separated"), ("Vedovo/a", "Widow")]
ABITAZIONE = [("Casa o appartamento", "House / apartment"), ("In affitto", "Rented apartment"), ("Con i genitori", "With parents"),
              ("Alloggio comunale", "Municipal apartment"), ("Alloggio aziendale", "Office apartment"), ("Cooperativa", "Co-op apartment")]
PROFESSIONI = [("Non indicata", "Non indicato")] + [(o, o) for o in cat["OCCUPATION_TYPE"] if o != "Non indicato"]


@zero_gpu
def valuta(reddito, eta, occupato, anzianita, figli, tipo_reddito, istruzione, stato, abitazione, professione,
           auto, casa, tel_lavoro, telefono, email):
    cliente = {
        "AMT_INCOME_TOTAL": float(reddito), "DAYS_BIRTH": -float(eta) * 365.25,
        "DAYS_EMPLOYED": -float(anzianita) * 365.25 if occupato else 365243,
        "CNT_CHILDREN": int(figli), "NAME_INCOME_TYPE": tipo_reddito, "NAME_EDUCATION_TYPE": istruzione,
        "NAME_FAMILY_STATUS": stato, "NAME_HOUSING_TYPE": abitazione,
        "OCCUPATION_TYPE": None if professione == "Non indicato" else professione,
        "FLAG_OWN_CAR": "Y" if auto else "N", "FLAG_OWN_REALTY": "Y" if casa else "N",
        "FLAG_WORK_PHONE": int(tel_lavoro), "FLAG_PHONE": int(telefono), "FLAG_EMAIL": int(email),
    }
    esito, p, motivi = scorer.valuta(cliente)
    icona = "✅" if esito == "IDONEO" else "❌"
    testo = f"## {icona} {esito}\n**Probabilità stimata di affidabilità: {p:.0%}**\n\n" + "\n".join(f"- {m}" for m in motivi)
    return testo


req = scorer.requisiti
with gr.Blocks(title="Credit scoring spiegabile") as demo:
    gr.Markdown(
        "# Credit scoring spiegabile\n"
        "Valutazione di una richiesta di carta di credito con un **albero decisionale** addestrato su 338 mila richieste. "
        "Ha le stesse prestazioni di una Random Forest (ROC AUC 0.978 contro 0.979), ma può motivare ogni esito. "
        f"Requisiti minimi ricavati dai dati: reddito ≥ {req['REDDITO']:,.0f}, età ≥ {req['ETA']:.0f} anni, "
        f"anzianità ≥ {req['ANZIANITA']:.1f} anni. Il genere non è usato dal modello.".replace(",", ".")
    )
    with gr.Row():
        with gr.Column():
            reddito = gr.Number(label="Reddito annuo", value=180000, minimum=0)
            eta = gr.Slider(18, 75, value=45, step=1, label="Età (anni)")
            occupato = gr.Checkbox(value=True, label="Attualmente occupato")
            anzianita = gr.Slider(0, 45, value=6, step=0.5, label="Anzianità lavorativa (anni)")
            figli = gr.Slider(0, 6, value=0, step=1, label="Figli")
        with gr.Column():
            tipo = gr.Dropdown(REDDITO_TIPO, value="Working", label="Tipo di reddito")
            istr = gr.Dropdown(ISTRUZIONE, value="Higher education", label="Istruzione")
            stato = gr.Dropdown(STATO_CIVILE, value="Married", label="Stato civile")
            abit = gr.Dropdown(ABITAZIONE, value="House / apartment", label="Abitazione")
            prof = gr.Dropdown(PROFESSIONI, value="Managers", label="Professione")
            with gr.Row():
                auto = gr.Checkbox(value=True, label="Auto di proprietà")
                casa = gr.Checkbox(value=True, label="Casa di proprietà")
            with gr.Row():
                tel_l = gr.Checkbox(label="Telefono di lavoro")
                tel = gr.Checkbox(value=True, label="Telefono fisso")
                mail = gr.Checkbox(value=True, label="Email")
    btn = gr.Button("Valuta la richiesta", variant="primary")
    out = gr.Markdown()
    inputs = [reddito, eta, occupato, anzianita, figli, tipo, istr, stato, abit, prof, auto, casa, tel_l, tel, mail]
    btn.click(valuta, inputs, out)
    gr.Markdown("Demo a scopo dimostrativo su un dataset didattico: non è uno strumento di valutazione del credito reale. "
                "I dati inseriti non vengono salvati.")

if __name__ == "__main__":
    demo.launch()
