import spacy
from spacy.tokens import Token
from negspacy.negation import Negex

if not Token.has_extension('negex'):
    Token.set_extension('negex', default=False)

nlp = spacy.load('en_core_web_sm')

# Add EntityRuler
ruler = nlp.add_pipe('entity_ruler', before='ner')
patterns = [{"label": "SYMPTOM", "pattern": "sore that won't heal"}]
ruler.add_patterns(patterns)

nlp.add_pipe('negex')

doc = nlp("I don't have a sore that won't heal, but I do have a new spot on my skin that is getting bigger.")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Negated: {ent._.negex}")
