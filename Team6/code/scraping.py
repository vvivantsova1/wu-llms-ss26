import requests
import polars as pl
import time
import re
import csv
import os
import random
from bs4 import BeautifulSoup

def fetch_real_cases_with_facts(num_cases=1, start_year=2026, end_year=2020, norm_gesucht="EStG"):
    url = "https://data.bka.gv.at/ris/api/v2.6/judikatur"
    #dataset = []

    HEADERS = {
        # Wir tarnen uns als Browser...
        #"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        # ...und hängen eine Info für den Admin an
        "From": "h12329053@s.wu.ac.at", 
        "X-Project": "University Data Science Course: Legal LLM Training - Scraping for Research"
    }
    
    # # Wir filtern nach EStG, um Steuerfälle zu bekommen
    # params = {
    #     "Applikation": "Vwgh",
    #     "Dokumenttyp": "Entscheidung",
    #     "Norm": "EStG", 
    #     "HitsPerPage": 50, 
    #     #"PageNumber": 1
    #     "Position": 1
    # }

    #backup_filename = "/home/sj5/Documents/linux_hanka_llm/finetuning_dataset_BACKUP.csv"
    backup_filename = "/mnt/red/red_hanka_bcthesis/llm/finetuning_dataset_ASVG_BACKUP.csv"
    # Falls die Datei noch nicht existiert, erstellen wir sie und schreiben den Header
    if not os.path.exists(backup_filename):
        with open(backup_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "instruction", "input", "output"])
    
    total_collected = 0
    current_year = start_year
    print(f"Starte Scraping für {norm_gesucht} von {start_year} bis {end_year}...")
    
    #while total_collected < num_cases:
    # ÄUßERE SCHLEIFE: Solange wir das Endjahr (z.B. 2020) noch nicht unterschritten haben
    while current_year >= end_year and total_collected < num_cases:
        page_number = 1
        # INNERE SCHLEIFE: Blättert durch alle Seiten des aktuellen Jahres
        while True:
            # --- DIE OFFIZIELLEN ÖSTERREICHISCHEN API PARAMETER ---
            params = {
                "Applikation": "Vwgh",
                "Dokumenttyp": "Entscheidung",
                "Norm": norm_gesucht, 
                "EntscheidungsdatumVon": f"{current_year}-01-01",
                "EntscheidungsdatumBis": f"{current_year}-12-31",
                "DokumenteProSeite": "OneHundred",  # "Ten", "Twenty", "Fifty" oder "OneHundred"
                "Seitennummer": page_number         # 1, 2, 3...
            }

            response = requests.get(url, params=params, headers=HEADERS)
            if response.status_code != 200:
                print(f"API Fehler: {response.status_code}")
                break

            if response.status_code == 429:
                print("Rate Limit erreicht! Pausiere für 5 Minuten...")
                time.sleep(300) # 5 Minuten schlafen
                continue
                
            data = response.json()
            
            raw_results = (
                data.get("OgdSearchResult", {})
                    .get("OgdDocumentResults", {})
                    .get("OgdDocumentReference", [])
            )

            # Wenn die API nur ein einzelnes Dict schickt, verpacken wir es selbst in eine Liste
            if isinstance(raw_results, dict):
                results = [raw_results]
            else:
                results = raw_results
            
            # Wenn das Jahr keine Ergebnisse mehr auf dieser Seite hat, 
            # brechen wir die innere Schleife ab und gehen zum nächsten Jahr!
            if not results:
                print(f"Jahr {current_year} ist komplett abgeschlossen. Keine weiteren Seiten.")
                break

            # --- ÄNDERUNG 2: Wir speichern nur die Fälle der AKTUELLEN Seite im RAM ---
            page_dataset = []
                
            for item in results:
                if not isinstance(item, dict):
                    continue

                d = item.get("Data", {})
                jud = d.get("Metadaten", {}).get("Judikatur", {})
                
                # 1. Das perfekte Label holen (Die Lösung)
                normen = jud.get("Normen", {}).get("item", "")
                if isinstance(normen, list):
                    norm = "; ".join([str(x) for x in normen if x])
                else:
                    norm = str(normen)
                    
                # 2. Den Link zum VOLLTEXT (JWT) holen
                # Das ist der Link, in dem die echte Geschichte steht!
                text_url = jud.get("EntscheidungstextUrl", "")
                
                if norm and text_url:
                    try:
                        # Wir laden das echte Urteil als HTML herunter
                        html_resp = requests.get(text_url, headers=HEADERS)
                        if html_resp.status_code == 200:
                            soup = BeautifulSoup(html_resp.text, 'html.parser')
                            
                            # Den gesamten Text der Webseite holen
                            full_text = soup.get_text(separator=' ', strip=True)
                            
                            # Den Müll am Anfang der Webseite (Menüs, Titel etc.) abschneiden.
                            # VwGH-Urteile beginnen inhaltlich meist nach dem Wort "Begründung" 
                            # oder "Entscheidungsgründe".
                            if "Begründung" in full_text:
                                facts_text = full_text.split("Begründung", 1)[-1]
                            elif "Sachverhalt" in full_text:
                                facts_text = full_text.split("Sachverhalt", 1)[-1]
                            else:
                                # Fallback: Wir überspringen die ersten 500 Zeichen Metadaten
                                facts_text = full_text[500:] 
                            
                            # Wir nehmen die ersten 2000 Zeichen der Begründung. 
                            # Dort erklärt das Gericht, was eigentlich passiert ist (Die Geschichte/Der Fall)
                            facts_clean = facts_text[:2000]
                            facts_clean = re.sub(r'\s+', ' ', facts_clean).strip()

                            case_id = d.get("Metadaten", {}).get("Technisch", {}).get("ID", "")
                                
                            # Als Liste speichern, damit der CSV-Writer es leicht schreiben kann
                            page_dataset.append([
                                case_id,
                                "Analysiere den folgenden steuerrechtlichen Sachverhalt und nenne die relevanten österreichischen Paragraphen.",
                                facts_clean,
                                norm
                            ])
                                                    
                            total_collected += 1
                            if total_collected >= num_cases:
                                break
                    except Exception as e:
                        pass

            # --- ÄNDERUNG 3: Sofortiges Speichern (Checkpoint) nach jeder Seite ---
            if page_dataset:
                with open(backup_filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(page_dataset)
            
            print(f"Jahr {current_year} | Seite {page_number} verarbeitet. Bisher gesamt: {total_collected} Fälle.")

            # Wenn wir genug Daten haben, brechen wir die innere Schleife ab
            if total_collected >= num_cases:
                break
            # Wir bereiten die nächste Seite für das aktuelle Jahr vor
            page_number += 1
            time.sleep(random.uniform(1.5, 3.5))
        
        # Nächstes Jahr vorbereiten
        current_year -= 1
    
    print("\nScraping abgeschlossen! Starte sichere Deduplikation...")

    if os.path.exists(backup_filename):
        # Wir laden das rohe, gesicherte Backup-CSV
        df = pl.read_csv(backup_filename)
        print("--- DEINE IDEE: SCHRITT 1 (Alles aggregieren) ---")
        # Polars wirft einfach alle Outputs eines Falles in eine Liste
        df_grouped = df.group_by(["instruction", "input"]).agg([
            pl.col("id").first(),
            pl.col("output") # Wird zu: ["EStG §16; EStG §20", "EStG §16; BAO §167"]
        ])
        print(df_grouped.select(["input", "output"]))


        # --- DEINE IDEE: SCHRITT 2 (Einzigartige Werte picken) ---
        # Diese pure Python-Funktion macht genau das, was du vorgeschlagen hast:
        # Sie nimmt die Liste, zerteilt alles beim Semikolon, entfernt Duplikate und klebt es zusammen.
        def clean_citations(citation_list):
            unique_laws = set()
            for item in citation_list:
                for law in item.split(";"):
                    unique_laws.add(law.strip()) # .strip() entfernt Leerzeichen
            return "; ".join(sorted(list(unique_laws)))

        # Wir wenden die Funktion sicher auf die Liste an.
        # try/except, da sich der Befehl in neueren Polars-Versionen von 'apply' zu 'map_elements' geändert hat.
        try:
            df_final = df_grouped.with_columns(
                pl.col("output").map_elements(clean_citations, return_dtype=pl.Utf8)
            )
        except AttributeError:
            # Für deine ältere Polars-Version
            df_final = df_grouped.with_columns(
                pl.col("output").apply(clean_citations, return_dtype=pl.Utf8)
            )

        print("\n--- SCHRITT 2: Finales Ergebnis ---")
        df_final = df_final.select(["id", "instruction", "input", "output"])
        print(df_final)

        df_final.write_csv("/mnt/red/red_hanka_bcthesis/llm/finetuning_ASVGdataset_1.csv")
        print(f"Deduplikation erfolgreich! {len(df_final)} einzigartige Fälle in 'finetuning_ASVGdataset_1.csv' gespeichert.")

#fetch_real_cases_with_facts(num_cases= 2000, start_year=2016, end_year=2000, norm_gesucht="EStG") or norm_gesucht="BAO" or norm_gesucht="UStG"
fetch_real_cases_with_facts(num_cases= 10000, start_year=2026, end_year=2000, norm_gesucht="ASVG")

#Things to change before running the code:
#1. norm_gesucht: "EStG", "BAO" oder "UStG" je nachdem, welche Norm du scrapen möchtest.
#2. num_cases: Wie viele Fälle du insgesamt sammeln möchtest. Je mehr, desto länger dauert es natürlich.
#3. start_year und end_year: Der Zeitraum, aus dem du Fälle sammeln möchtest. Je größer der Zeitraum, desto mehr Fälle kannst du potenziell bekommen, aber es dauert auch länger.
#4. backup_filename: Der Pfad, unter dem du die gesammelten Daten zwischenspeichern möchtest. Achte darauf, dass du Schreibrechte hast und genug Speicherplatz vorhanden ist.
#5. filename in df_final.write_csv(): Der Pfad, unter dem die finale, deduplizierte CSV gespeichert wird. Auch hier solltest du Schreibrechte haben und genug Speicherplatz bereitstellen.
#6. HEADERS: Du kannst die User-Agent-Zeile an deine Email Adresse anpassen, damit der Admin der API weiß, wer da gerade Daten abruft. Das ist höflich und kann helfen, falls es Probleme gibt.
