import pandas as pd
from sklearn.metrics import classification_report
import os
import sys

# Ensure Python can find the pipeline module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.pipeline import run_pipeline

def evaluate(records):
    # records: list of { 'text': str, 'true_disease': str }
    y_true, y_pred = [], []
    
    for rec in records:
        res = run_pipeline(rec['text'])
        top = res['ranked_diseases'][0]
        
        if top['confidence_level'] == 'NO INDICATION':
            y_pred.append('NO INDICATION')
        else:
            y_pred.append(top['disease'])
            
        y_true.append(rec['true_disease'])
        
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

def run_evaluation(csv_path):
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')
    
    print(f"Evaluating {len(records)} records...")
    report = evaluate(records)
    
    print("\n--- Classification Report ---")
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    
    # As per spec: Be honest about errors [cite: 254]
    print("\n--- Error Analysis (Misclassifications) ---")
    for rec in records:
        res = run_pipeline(rec['text'])
        top = res['ranked_diseases'][0]
        pred = top['disease'] if top['confidence_level'] != 'NO INDICATION' else 'NO INDICATION'
        
        if pred != rec['true_disease']:
            print(f"FAILED: Expected '{rec['true_disease']}', Got '{pred}'")
            print(f"Input Text: {rec['text']}")
            print(f"Weights Used: {top['fusion_weights']}\n")

if __name__ == "__main__":
    records_file = os.path.join(os.path.dirname(__file__), 'records.csv')
    if os.path.exists(records_file):
        run_evaluation(records_file)
    else:
        print(f"Error: Could not find {records_file}. Please create it with 'text' and 'true_disease' columns.")