DESCRIPTION:
- realisation d'un script d'inférence générique:
- n'importe quelle modele de text recognition de doctr
- proposition de plusieurs solutions de labels (top K labels d'entropie la plus basse)
- utiliser tensorflow pour le moment

METHODE:
- reprise du srcipt d'inference de doctr  https://github.com/mindee/doctr/blob/main/references/recognition/evaluate_tensorflow.py
- adaptation du postprocessor pour generer les sequences differentes depuis les logits
- fournir les logits et effectuer des modeifications pour determiner les sequences puis appliquer le model.postprocessor(logits_modified)
