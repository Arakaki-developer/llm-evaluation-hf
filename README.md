# Evaluating Text Summarization Models with HuggingFace Evaluate
This project demonstrates how to use the `evaluate` library from HuggingFace to compare the performance of different text summarization models.

- Loads sample texts and references
- Generates summaries with multiple models
- Computes ROUGE and BLEU scores

## How to use

1. Install requirements  
   `pip install -r requirements.txt`
2. Run the script  
   `python summarize_and_evaluate.py`

## Results

### Results for model: facebook/bart-large-cnn
ROUGE: {'rouge1': np.float64(0.5566), 'rouge2': np.float64(0.3334), 'rougeL': np.float64(0.4838), 'rougeLsum': np.float64(0.4861)}
BLEU: 0.2118

### Results for model: google/pegasus-xsum
ROUGE: {'rouge1': np.float64(0.2149), 'rouge2': np.float64(0.0732), 'rougeL': np.float64(0.1709), 'rougeLsum': np.float64(0.1726)}
BLEU: 0.0507
