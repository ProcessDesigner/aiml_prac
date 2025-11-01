import scispacy
import spacy
#Core models
import en_core_sci_sm
import en_core_sci_md
#NER specific models
import en_ner_bc5cdr_md # extracting disease and drugs
#Tools for extracting & displaying data
from spacy import displacy
import pandas as pd

mtsample_df=pd.read_csv('/content/mtsamples.csv')
mtsample_df.head()

# Pick specific transcription to use (row 3, column "transcription") and test the scispacy NER model
text = mtsample_df.loc[10, "transcription"]

nlp_sm = en_core_sci_sm.load()
doc = nlp_sm(text)
#Display resulting
#entity extraction
displacy_image = displacy.render(doc, jupyter=True,style='ent')

# Note the entity is tagged here. Mostly medicalterms. However, these are generic entities.



nlp_md = en_core_sci_md.load()
doc = nlp_md(text)
#Display resulting entity extraction
displacy_image = displacy.render(doc, jupyter=True,style='ent')

# This time the numbers are also tagged as entities by en_core_sci_md.

# Now Load specific model: import en_ner_bc5cdr_md and pass text through

# A spaCy NER model trained on the BC5CDR corpus( the main part or body of a bodily structure or organ)

# BC5CDR corpus consists of 1500 PubMed articles with 4409 annotated chemicals, 5818 diseases and 3116 chemical-disease interactions.


# used to DISEASE, CHEMICAL

nlp_bcc = en_ner_bc5cdr_md.load()
doc = nlp_bcc(text)
#Display resulting entity extraction
displacy_image = displacy.render(doc, jupyter=True,style='ent')

doc = nlp_bcc(text)
print(doc.ents)
print("TEXT", "START", "END", "ENTITY TYPE")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

mtsample_df.dropna(subset=['transcription'], inplace=True)
mtsample_df_subset = mtsample_df.sample(n=100, replace=False, random_state=42)   #replacebool, default False :disallow sampling of the same row more than once.
mtsample_df_subset.info()
mtsample_df_subset.head()

# spaCy matcher – The rule-based matching resembles the usage of regular expressions, but spaCy provides additional capabilities. Using the tokens and relationships within a document enables you to identify patterns that include entities with the help of NER models. The goal is to locate drug names and their dosages from the text, which could help detect medication errors by comparing them with standards and guidelines.

# The goal is to locate drug names and their dosages from the text, which could help detect medication errors by comparing them with standards and guidelines.

from spacy.matcher import Matcher
pattern = [{'ENT_TYPE':'CHEMICAL'}, {'LIKE_NUM': True}, {'IS_ASCII': True}]
matcher = Matcher(nlp_bcc.vocab)
matcher.add("DRUG_DOSE", [pattern])

# The code above creates a pattern to identify a sequence of three tokens:

# A token whose entity type is CHEMICAL (drug name)

# A token that resembles a number (dosage)

# A token that consists of ASCII characters (units, like mg or mL)

# Then we initialize the Matcher with a vocabulary. The matcher must always share the same vocab with the documents it will operate on, so we use the nlp_bcc object vocab. We then add this pattern to the matcher and give it an ID.

for transcription in mtsample_df_subset['transcription']:
    doc = nlp_bcc(transcription)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp_bcc.vocab.strings[match_id]  # get string representation
        span = doc[start:end]  # the matched span adding drugs doses
        print(span.text, start, end, string_id,)
    
    
#Now we can loop through all transcriptions and extract the text matching this pattern:
for transcription in mtsample_df_subset['transcription']:
    doc = nlp_bcc(transcription)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp_bcc.vocab.strings[match_id]  # get string representation
        span = doc[start:end]  # the matched span adding drugs doses
        print(span.text, start, end, string_id,)
        #Add disease and chemical
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

