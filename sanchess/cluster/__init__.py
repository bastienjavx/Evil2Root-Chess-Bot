"""San-o1 — couche de calcul distribué bénévole (style Leela Chess Zero).

Trois rôles, asynchrones et tolérants aux pannes (nœuds non fiables sur Internet) :

  - `server`  : COORDINATEUR (FastAPI, SANS torch) déployé sur Railway. Distribue les
                jobs de self-play, collecte/valide les parties, sert le modèle courant
                (octets opaques) et un dashboard public + leaderboard. État persistant
                sur un Volume Railway (modèle + buffer de shards + sqlite).
  - `worker`  : CLIENT bénévole. Auto-détecte CUDA/MPS/CPU, télécharge le modèle, joue
                du self-play (réutilise train/selfplay*.py) et renvoie les parties.
  - `trainer` : CONNECTEUR sur une machine GPU. Tire les parties, lance train/online.py
                (inchangé) pour les pas de gradient, et republie le modèle amélioré.

Le coordinateur ne charge JAMAIS torch : il ne fait que router des octets et valider
des shards de samples avec python-chess. Voir CLUSTER.md pour le déploiement.
"""
