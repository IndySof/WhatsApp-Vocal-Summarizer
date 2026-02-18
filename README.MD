# WhatsApp Vocal Summarizer — Because Your Girlfriend Deserves Sleep

> *"Ma meilleure amie m'a envoyé 24 vocaux en une nuit. Je vais y passer 8 heures !!"*
> — Le plus gros problème de ma petite copine

---

## Le problème

Ta petite copine a une meilleure amie. Elle est merveilleuse. Elle est expressive. Elle a des opinions sur tout, la mort du chat du voisin ou un film romantique nul regardé à 2h du matin.

Et elle communique exclusivement par **vocaux WhatsApp de 3 à 24 minutes**.

En une seule soirée, ta copine peut recevoir jusqu'à 24 vocaux représentant près de **2 heures d'audio non-stop**.

Ta copine dormait.

Toi, tu veux qu'elle soit heureuse ET qu'elle s'occupe de toi (accessoirement).

**Ce programme est la solution.**

---

## Ce que fait `main.py`

`main.py` prend l'intégralité des fichiers `.opus` reçus sur WhatsApp, les transcrit un par un grâce à **Groq Whisper Large V3**, puis génère automatiquement un **résumé structuré** via **LLaMA 3.3 70B** pour que ta copine puisse répondre intelligemment à sa meilleure amie sans avoir écouté 2 heures d'audio.

Le flux est simple :

```
24 vocaux de 3 min chacun
        ↓
   Whisper écoute tout
        ↓
  LLaMA résume tout
        ↓
  Ta copine lit en 3 min
        ↓
  Ta copine répond comme si elle avait tout écouté
        ↓
  L'amitié est sauvée. Tout le monde dort.
```

Personne n'a été blessé dans l'opération.

---

## Prérequis

- Python 3.8+
- Une clé API Groq (gratuite sur [console.groq.com](https://console.groq.com))
- Des vocaux `.opus` exportés depuis WhatsApp
- De la patience (pour les 4 premières secondes, après c'est automatique)

---

## Installation

```bash
pip install groq
```

C'est tout. On n'est pas là pour souffrir davantage.

---

## Configuration

Ouvre `main.py` et remplace la clé API en haut du fichier :

```python
GROQ_API_KEY = "ta_clé_groq_ici"
LANGUAGE = "fr"  # ou "it", "en", selon la langue des vocaux
```

---

## Utilisation

**Étape 1 — Exporte les vocaux WhatsApp**

Sur WhatsApp : ouvre la conversation → `...` → *Exporter la discussion* → *Avec médias*

Extrais l'archive. Tu obtiendras des fichiers `.opus`.

**Étape 2 — Lance le script**

```bash
python main.py /chemin/vers/le/dossier/des/vocaux
```

Ou si tu es déjà dans le dossier :

```bash
python main.py .
```

**Étape 3 — Va te faire un café**

Le script tourne environ 2 à 3 minutes selon le nombre de fichiers. Tu le mérites.

**Étape 4 — Lis les résultats**

Deux fichiers sont générés automatiquement :

| Fichier | Contenu |
|---|---|
| `transcriptions.json` | Toutes les transcriptions brutes, une entrée par vocal |
| `resume.txt` | Résumé complet : thèmes abordés, points clés, actions à prendre, ton général |

---

## Exemple de résumé généré

```
1. De quoi parle-t-on globalement ?
   Un dimanche après-midi raconté en 31 vocaux : une dispute avec
   la caissière de Lidl, une théorie sur les dauphins, et une remise
   en question existentielle causée par un yaourt périmé.

2. Points importants
   - A failli acheter une plante mais ne l'a pas fait (15 min de vocal)
   - Pense que son voisin du dessus est peut-être un robot
   - A regardé 4 épisodes d'une série qu'elle déteste "pour voir si ça s'améliore"
   - Le yaourt était périmé depuis janvier. On est en octobre.

3. Actions demandées
   - Confirmer que les dauphins sont effectivement louches
   - Lui dire si elle devrait rappeler le voisin robot ou pas
   - Ne surtout pas lui dire qu'elle a tort sur le yaourt

4. Ton général
   Chaos organisé. Commence chaque vocal par "bon alors" et le finit
   par "bref de toute façon". Parle du yaourt avec le même sérieux
   qu'une conférence de l'ONU.
```

---

## Limites connues

**Rate limit Groq** — Le plan gratuit permet 7200 secondes d'audio par heure. Si ton amie a été particulièrement prolifica ce soir-là, le script s'arrête et te dit d'attendre quelques minutes avant de relancer. C'est elle le problème, pas le code.

**Qualité de transcription** — Whisper est excellent mais peut trébucher sur les mélanges de langues, les mots inventés, et les références à des idols coréens prononcés avec un accent napolitain.

**Ce programme ne résout pas le problème de fond** — Ton amie enverra toujours 24 vocaux par nuit. Mais au moins, ta copine pourra y répondre.

---

## Stack technique

| Composant | Technologie |
|---|---|
| Transcription audio | Groq Whisper Large V3 |
| Résumé intelligent | LLaMA 3.3 70B Versatile |
| Format audio supporté | `.opus` (WhatsApp natif) |
| Langage | Python 3.8+ |

---

## Contribuer

Si toi aussi tu as une amie qui envoie des vocaux de 8 minutes à 1h du matin, tu es le bienvenu pour contribuer. Les PR sont ouvertes.

Fonctionnalités souhaitées :
- Détection automatique de la langue
- Génération automatique d'une réponse dans le style de la meilleure amie
- Alerte sonore quand le résumé est prêt pour ne pas avoir à regarder le terminal à 2h du matin

---

## Licence

MIT — Fais-en ce que tu veux, mais si ça sauve une amitié, envoie-nous un message.

---

*Fait avec amour, Groq, et une légère incompréhension du concept de "message court".*