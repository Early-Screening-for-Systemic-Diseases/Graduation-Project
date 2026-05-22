import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("I don't have a sore that won't heal, but I do have a new spot on my skin that is getting bigger.")
for token in doc:
    if token.dep_ == 'neg':
        print(f"Negation: {token.text}")
        print("Scope lemmas:")
        print([t.lemma_ for t in token.head.subtree])
