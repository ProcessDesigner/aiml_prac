# ---------------------------------------------------------------
# Experiment 5: Extracting Medical Information From Clinical Text With NLP
# ---------------------------------------------------------------

# -------------------------
# 1. Install required packages
# -------------------------
# Run these only once in your environment (uncomment if running first time)
# !pip install spacy scispacy
# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz
# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz
# !pip install pandas numpy

# -------------------------
# 2. Import libraries
# -------------------------
import spacy
import scispacy
import pandas as pd
import numpy as np

# -------------------------
# 3. Read the data
# -------------------------
# Dataset: mtsamples.csv — contains medical transcription text data
# Make sure the CSV file is in your working directory
df = pd.read_csv("mtsamples.csv")

# Display first few rows
print("Dataset Loaded Successfully!")
print(df.head())

# -------------------------
# 4. Drop missing values
# -------------------------
df.dropna(subset=["description"], inplace=True)
print(f"\nNumber of records after dropping NaN: {len(df)}")

# -------------------------
# 5. Create a smaller sample for quick testing
# -------------------------
sample_df = df.sample(5, random_state=42)
texts = sample_df["description"].tolist()

# -------------------------
# 6. Load and test SpaCy models
# -------------------------

# (a) Load small SciSpaCy model
print("\n--- Testing en_core_sci_sm model ---")
nlp_small = spacy.load("en_core_sci_sm")
for text in texts[:2]:
    doc = nlp_small(text)
    print(f"\nText: {text[:150]}...")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

# (b) Load medium SciSpaCy model
print("\n--- Testing en_core_sci_md model ---")
nlp_medium = spacy.load("en_core_sci_md")
for text in texts[:2]:
    doc = nlp_medium(text)
    print(f"\nText: {text[:150]}...")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

# (c) Load biomedical NER model
print("\n--- Testing en_ner_bc5cdr_md (Disease & Chemical Named Entity Recognition) ---")
nlp_ner = spacy.load("en_ner_bc5cdr_md")

# Example clinical text
example_text = """
The patient was prescribed 500mg of Metformin for Type 2 Diabetes.
He also reported a mild allergy to Penicillin.
"""
doc = nlp_ner(example_text)

print("\nEntities Extracted using en_ner_bc5cdr_md:")
for ent in doc.ents:
    print(f"Text: {ent.text} | Label: {ent.label_}")

# -------------------------
# 7. Apply the NER model to the sample dataset
# -------------------------
print("\n--- Extracting Entities from Sample Data ---")
for i, text in enumerate(texts):
    doc = nlp_ner(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"\nSample {i+1}:")
    print("Entities:", entities)

# -------------------------
# 8. Summary
# -------------------------
print("\n--- Summary ---")
print("✔ Extracted key medical entities such as diseases, drugs, and chemicals.")
print("✔ Compared performance across multiple SciSpaCy models.")
print("✔ Demonstrated how NLP can extract structured medical data from unstructured clinical text.")
