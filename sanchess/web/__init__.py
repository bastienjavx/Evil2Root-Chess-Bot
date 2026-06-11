"""Interface web San-o1 : API REST + WebSocket pour jouer contre le modèle en
direct, suivre l'entraînement, et inspecter les modèles/checkpoints exposés.

Lancement :  python -m sanchess.web   (ou ./scripts/run_web.sh)
Exposition publique :  cloudflared tunnel --url http://localhost:8000
"""
