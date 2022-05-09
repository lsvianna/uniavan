%%capture
!pip install spacy

import spacy
model = spacy.load('/content/drive/MyDrive/informatica_saude/model')

model('Quem não vê apenas o lado positivo dos outros cria um inferno para si próprio.').cats
