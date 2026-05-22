import spacy
from spacy.tokens import Token
from negspacy.negation import Negex

if not Token.has_extension('negex'):
    Token.set_extension('negex', default=False)

nlp = spacy.load('en_core_web_sm')

# Build lemma-based pattern
symptom_text = "sore that won't heal"
lemma_pattern = [{"LEMMA": t.lemma_.lower()} for t in nlp(symptom_text) if not t.is_punct]
patterns = [{"label": "SYMPTOM", "pattern": lemma_pattern, "id": symptom_text}]

ruler = nlp.add_pipe('entity_ruler', before='ner')
ruler.add_patterns(patterns)
nlp.add_pipe('negex')

# The text has "will not" instead of "won't", to simulate contractions.fix
doc = nlp("I do not have a sore that will not heal, but I do have a new spot on my skin that is getting bigger.")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Negated: {ent._.negex}")
