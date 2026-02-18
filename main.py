#!/usr/bin/env python3
"""
Transcription et résumé de vocaux WhatsApp via Groq (Whisper + LLaMA)
Usage: python transcribe_whatsapp.py [dossier_audio]
       Si aucun dossier spécifié, utilise le dossier courant.
"""

import os
import sys
import glob
import json
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = "YOURKEYHERE"
WHISPER_MODEL = "whisper-large-v3"
SUMMARY_MODEL = "llama-3.3-70b-versatile"
LANGUAGE = "fr"
# ──────────────────────────────────────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

def transcribe_file(filepath):
    """Transcrit un fichier audio .opus via Groq Whisper."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(filename, f, "audio/opus"),
            model=WHISPER_MODEL,
            language=LANGUAGE,
            response_format="text"
        )
    return result.strip()

def summarize(transcriptions):
    """Envoie toutes les transcriptions à LLaMA pour un résumé structuré."""
    numbered = "\n\n".join(
        f"[Message {i+1}] {t['text']}"
        for i, t in enumerate(transcriptions)
        if t["text"] and not t["text"].startswith("ERREUR")
    )

    prompt = f"""Voici la transcription de {len(transcriptions)} messages vocaux WhatsApp reçus dans l'ordre chronologique.

{numbered}

Fais-moi un résumé clair et structuré en français :
1. De quoi parle-t-on globalement ?
2. Les points importants / informations clés
3. Les actions demandées ou décisions à prendre (si applicable)
4. Le ton général de la conversation
"""

    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(folder, "*.opus")))

    if not files:
        print(f"❌ Aucun fichier .opus trouvé dans : {os.path.abspath(folder)}")
        sys.exit(1)

    print(f"📁 {len(files)} fichiers audio trouvés dans : {os.path.abspath(folder)}")
    print("=" * 60)

    transcriptions = []

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"🎙️  [{i+1}/{len(files)}] {filename}")
        try:
            text = transcribe_file(filepath)
            transcriptions.append({"file": filename, "text": text})
            print(f"    ✅ {text[:120]}{'...' if len(text) > 120 else ''}")
        except Exception as e:
            print(f"    ❌ Erreur : {e}")
            transcriptions.append({"file": filename, "text": f"ERREUR: {e}"})

    # Sauvegarde des transcriptions brutes
    output_json = "transcriptions.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Transcriptions sauvegardées dans : {output_json}")

    # Résumé global
    print("\n" + "=" * 60)
    print("🤖 Génération du résumé en cours...")
    print("=" * 60)
    try:
        summary = summarize(transcriptions)
        print("\n" + summary)

        output_txt = "resume.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("RÉSUMÉ DES VOCAUX WHATSAPP\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("TRANSCRIPTIONS DÉTAILLÉES\n")
            f.write("=" * 60 + "\n\n")
            for i, t in enumerate(transcriptions):
                f.write(f"[Message {i+1}] {t['file']}\n")
                f.write(t['text'] + "\n\n")

        print(f"\n💾 Résumé complet sauvegardé dans : {output_txt}")

    except Exception as e:
        print(f"❌ Erreur lors du résumé : {e}")

if __name__ == "__main__":
    main()