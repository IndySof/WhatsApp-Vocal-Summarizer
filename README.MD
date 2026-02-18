# WhatsApp Vocal Summarizer — Because Your Girlfriend Deserves Sleep
# Résumeur de Vocaux WhatsApp — Parce que ta copine mérite de dormir

> *"Ma meilleure amie m'a envoyé 24 vocaux en une nuit. Je vais y passer 8 heures !!"*
> — Le plus gros problème de ma petite copine

> *"My best friend sent me 24 voice messages in one night. I'll spend 8 hours on this!!"*
> — My girlfriend's biggest problem

---

## 🇫🇷 Le problème | 🇬🇧 The Problem

**FR:** Ta petite copine a une meilleure amie. Elle est merveilleuse. Elle est expressive. Elle a des opinions sur tout, la mort du chat du voisin ou un film romantique nul regardé à 2h du matin.

Et elle communique exclusivement par **vocaux WhatsApp de 3 à 24 minutes**.

En une seule soirée, ta copine peut recevoir jusqu'à 24 vocaux représentant près de **2 heures d'audio non-stop**.

Ta copine dormait.

Toi, tu veux qu'elle soit heureuse ET qu'elle s'occupe de toi (accessoirement).

**Ce programme est la solution.**

---

**EN:** Your girlfriend has a best friend. She's wonderful. She's expressive. She has opinions about everything, from the neighbor's dead cat to a terrible romantic movie watched at 2 AM.

And she communicates exclusively through **WhatsApp voice messages lasting 3 to 24 minutes**.

In a single evening, your girlfriend can receive up to 24 voice messages representing nearly **2 hours of non-stop audio**.

Your girlfriend was sleeping.

You want her to be happy AND to pay attention to you (incidentally).

**This program is the solution.**

---

## 🚀 Ce que fait `main.py` | What `main.py` Does

**FR:** `main.py` prend l'intégralité des fichiers `.opus` reçus sur WhatsApp, les transcrit un par un grâce à **Groq Whisper Large V3**, puis génère automatiquement un **résumé structuré** via **LLaMA 3.3 70B** pour que ta copine puisse répondre intelligemment à sa meilleure amie sans avoir écouté 2 heures d'audio.

**EN:** `main.py` takes all `.opus` files received on WhatsApp, transcribes them one by one using **Groq Whisper Large V3**, then automatically generates a **structured summary** via **LLaMA 3.3 70B** so your girlfriend can intelligently respond to her best friend without listening to 2 hours of audio.

Le flux est simple | The flow is simple:
```
24 vocaux de 3 min chacun | 24 voice messages of 3 min each
        ↓
   Whisper écoute tout | Whisper listens to everything
        ↓
  LLaMA résume tout | LLaMA summarizes everything
        ↓
  Ta copine lit en 3 min | Your girlfriend reads in 3 min
        ↓
  Ta copine répond comme si elle avait tout écouté
  Your girlfriend responds as if she listened to everything
        ↓
  L'amitié est sauvée. Tout le monde dort.
  Friendship is saved. Everyone sleeps.
```

Personne n'a été blessé dans l'opération | No one was harmed in this operation.

---

## 📋 Prérequis | Prerequisites

- Python 3.8+
- 🇫🇷 Une clé API Groq (gratuite sur [console.groq.com](https://console.groq.com))
- 🇬🇧 A Groq API key (free at [console.groq.com](https://console.groq.com))
- 🇫🇷 Des vocaux `.opus` exportés depuis WhatsApp
- 🇬🇧 `.opus` voice messages exported from WhatsApp
- 🇫🇷 De la patience (pour les 4 premières secondes, après c'est automatique)
- 🇬🇧 Patience (for the first 4 seconds, then it's automatic)

---

## 💾 Installation
```bash
pip install groq python-dotenv
```

🇫🇷 C'est tout. On n'est pas là pour souffrir davantage.

🇬🇧 That's it. We're not here to suffer more.

---

## ⚙️ Configuration

**FR:** Créez un fichier `.env` à la racine du projet :

**EN:** Create a `.env` file at the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
LANGUAGE=fr  # or "it", "en", depending on voice message language
```

---

## 🎯 Utilisation | Usage

### **Étape 1 | Step 1 — Exporte les vocaux WhatsApp | Export WhatsApp Voice Messages**

**FR:** Sur WhatsApp : ouvre la conversation → `...` → *Exporter la discussion* → *Avec médias*

Extrais l'archive. Tu obtiendras des fichiers `.opus`.

**EN:** On WhatsApp: open the conversation → `...` → *Export chat* → *With media*

Extract the archive. You'll get `.opus` files.

---

### **Étape 2 | Step 2 — Lance le script | Run the Script**
```bash
python main.py /path/to/voice/messages/folder
```

🇫🇷 Ou si tu es déjà dans le dossier :

🇬🇧 Or if you're already in the folder:
```bash
python main.py .
```

---

### **Étape 3 | Step 3 — Va te faire un café | Get a Coffee**

**FR:** Le script tourne environ 2 à 3 minutes selon le nombre de fichiers. Tu le mérites.

**EN:** The script runs for about 2 to 3 minutes depending on the number of files. You deserve it.

---

### **Étape 4 | Step 4 — Lis les résultats | Read the Results**

**FR:** Deux fichiers sont générés automatiquement :

**EN:** Two files are automatically generated:

| Fichier | File | Contenu | Content |
|---|---|---|---|
| `transcriptions.json` | `transcriptions.json` | Toutes les transcriptions brutes | All raw transcriptions |
| `resume.txt` | `summary.txt` | Résumé complet | Complete summary |

---

## 📝 Exemple de résumé généré | Example Generated Summary
```
1. De quoi parle-t-on globalement ? | What are we talking about overall?
   Un dimanche après-midi raconté en 31 vocaux : une dispute avec
   la caissière de Lidl, une théorie sur les dauphins, et une remise
   en question existentielle causée par un yaourt périmé.
   
   A Sunday afternoon told in 31 voice messages: an argument with
   the Lidl cashier, a theory about dolphins, and an existential
   questioning caused by expired yogurt.

2. Points importants | Important Points
   - A failli acheter une plante mais ne l'a pas fait (15 min de vocal)
     Almost bought a plant but didn't (15 min voice message)
   - Pense que son voisin du dessus est peut-être un robot
     Thinks her upstairs neighbor might be a robot
   - A regardé 4 épisodes d'une série qu'elle déteste "pour voir si ça s'améliore"
     Watched 4 episodes of a series she hates "to see if it gets better"
   - Le yaourt était périmé depuis janvier. On est en octobre.
     The yogurt expired in January. It's October.

3. Actions demandées | Requested Actions
   - Confirmer que les dauphins sont effectivement louches
     Confirm that dolphins are indeed suspicious
   - Lui dire si elle devrait rappeler le voisin robot ou pas
     Tell her whether she should call the robot neighbor back or not
   - Ne surtout pas lui dire qu'elle a tort sur le yaourt
     Definitely don't tell her she's wrong about the yogurt

4. Ton général | General Tone
   Chaos organisé. Commence chaque vocal par "bon alors" et le finit
   par "bref de toute façon". Parle du yaourt avec le même sérieux
   qu'une conférence de l'ONU.
   
   Organized chaos. Starts each voice message with "so anyway" and ends
   with "whatever anyway". Talks about yogurt with the same seriousness
   as a UN conference.
```

---

## ⚠️ Limites connues | Known Limitations

**🇫🇷 Rate limit Groq** — Le plan gratuit permet 7200 secondes d'audio par heure. Si ton amie a été particulièrement prolifica ce soir-là, le script s'arrête et te dit d'attendre quelques minutes avant de relancer. C'est elle le problème, pas le code.

**🇬🇧 Groq Rate Limit** — The free plan allows 7200 seconds of audio per hour. If your friend was particularly prolific that night, the script stops and tells you to wait a few minutes before restarting. She's the problem, not the code.

---

**🇫🇷 Qualité de transcription** — Whisper est excellent mais peut trébucher sur les mélanges de langues, les mots inventés, et les références à des idols coréens prononcés avec un accent napolitain.

**🇬🇧 Transcription Quality** — Whisper is excellent but may stumble on language mixing, made-up words, and references to Korean idols pronounced with a Neapolitan accent.

---

**🇫🇷 Ce programme ne résout pas le problème de fond** — Ton amie enverra toujours 24 vocaux par nuit. Mais au moins, ta copine pourra y répondre.

**🇬🇧 This program doesn't solve the root problem** — Your friend will still send 24 voice messages per night. But at least your girlfriend can respond to them.

---

## 🛠️ Stack technique | Tech Stack

| Composant | Component | Technologie | Technology |
|---|---|---|---|
| Transcription audio | Audio Transcription | Groq Whisper Large V3 | Groq Whisper Large V3 |
| Résumé intelligent | Smart Summary | LLaMA 3.3 70B Versatile | LLaMA 3.3 70B Versatile |
| Format audio | Audio Format | `.opus` (WhatsApp natif) | `.opus` (WhatsApp native) |
| Langage | Language | Python 3.8+ | Python 3.8+ |

---

## 🤝 Contribuer | Contributing

**FR:** Si toi aussi tu as une amie qui envoie des vocaux de 8 minutes à 1h du matin, tu es le bienvenu pour contribuer. Les PR sont ouvertes.

**EN:** If you also have a friend who sends 8-minute voice messages at 1 AM, you're welcome to contribute. PRs are open.

**Fonctionnalités souhaitées | Desired Features:**
- 🇫🇷 Détection automatique de la langue | 🇬🇧 Automatic language detection
- 🇫🇷 Génération automatique d'une réponse dans le style de la meilleure amie | 🇬🇧 Automatic response generation in the best friend's style
- 🇫🇷 Alerte sonore quand le résumé est prêt | 🇬🇧 Sound alert when summary is ready

---

## 📄 Licence | License

MIT — 🇫🇷 Fais-en ce que tu veux, mais si ça sauve une amitié, envoie-nous un message.

MIT — 🇬🇧 Do what you want with it, but if it saves a friendship, send us a message.

---

*🇫🇷 Fait avec amour, Groq, et une légère incompréhension du concept de "message court".*

*🇬🇧 Made with love, Groq, and a slight misunderstanding of the "short message" concept.*
