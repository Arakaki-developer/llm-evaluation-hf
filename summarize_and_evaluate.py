#Imports
import pandas as pd
from transformers import pipeline
import evaluate

# Carrega exemplo
df = pd.read_csv("sample_data.csv")
texts = df["text"].tolist() # Lista de texto
references = df["reference_summary"].tolist() # Lista de resumo 

# Modelos
model_names = [
    "facebook/bart-large-cnn",
    "google/pegasus-xsum"
]
summarizers = [pipeline("summarization", model=mn) for mn in model_names]

# Para cada modelo gera o resumo
all_generated = []
for summarizer in summarizers:
    summaries = []
    for t in texts:
        summary = summarizer(t, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
        summaries.append(summary)
    all_generated.append(summaries)

# Evaluation: ROUGE e BLEU
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Mostra os resultados
for idx, model_name in enumerate(model_names):
    print(f"\nResults for model: {model_name}")
    rouge_result = rouge.compute(predictions=all_generated[idx], references=references)
    bleu_result = bleu.compute(predictions=all_generated[idx], references=references)
    print("ROUGE:", {k: round(v,4) for k,v in rouge_result.items()})
    print("BLEU:", round(bleu_result["bleu"], 4))
    print("----")
