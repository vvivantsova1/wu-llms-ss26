import os

# 1. Environment & Storage Setup
# Redirect Hugging Face cache to your mounted Windows SSD
os.environ["HF_HOME"] = "/mnt/windows/windows_hanka_bcthesis/llm/huggingface_cache"
# Isolate the RTX 3080 (Assuming it is GPU 0. Change to "1" if it's your second slot)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
#os.environ["BNB_CUDA_VERSION"] = "121"

import polars as pl
import csv
from tqdm import tqdm
# Now we can safely import the deep learning libraries
from unsloth import FastLanguageModel
import torch
import time
import anthropic
from dotenv import load_dotenv

import ast
import faiss
#import requests
#import re
#import json
#import pickle
from pathlib import Path
#from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from typing import List, Dict
import numpy as np

class mistral:
    def __init__(self, model_name="mistral-7b-instruct-v0.1.Q4_0.gguf", device="cuda:0"):
        #self.model = FastLanguageModel(model_name, device=device)
        model_name = model_name        

    # 1. Loading Function
    def load_model_and_tokenizer(self, max_seq_length, dtype, load_in_4bit, model_name):
        """Loads the model and tokenizer once, and prepares them for fast inference."""
        print(f"Loading {model_name}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        # Enable Unsloth's native 2x faster inference mode
        FastLanguageModel.for_inference(model)
        
        # Return BOTH so the rest of the script can use them
        return model, tokenizer

    # 2. Inference Loop Function
    def generate_mistral_baseline(self, input_csv_path, backup_csv_path, prompt_template, model, tokenizer, limit=200):
        """Reads dataset, generates predictions, and saves line-by-line."""
        print(f"Loading data from {input_csv_path}...")
        df = pl.read_csv(input_csv_path)
        
        if limit:
            df = df.head(limit)
            
        start_index = 0
        
        if os.path.exists(backup_csv_path):
            existing_df = pl.read_csv(backup_csv_path)
            start_index = existing_df.height
            print(f"Found existing backup. Resuming from case {start_index}...")
            if start_index >= df.height:
                print("Baseline generation already complete!")
                return
        else:
            print("Starting fresh baseline generation...")
            with open(backup_csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "instruction", "input", "ground_truth_label", "raw_model_response"])

        for i in tqdm(range(start_index, df.height), desc="Generating Predictions"):
            row = df.row(i, named=True)
            
            # 1. Format the strict prompt
            prompt = prompt_template.format(row["instruction"], row["input"])
            
            # WE NOW HAVE THE TOKENIZER HERE!
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            
            # 2. Generate
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id 
            )
            
            # 3. Decode and Extract
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            final_answer = response.split("### Response:\n")[-1].strip()
            
            # 4. Save immediately to disk
            with open(backup_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([row["id"], row["instruction"], row["input"], row["output"], final_answer])

        print(f"\n Mistral baseline complete! Results saved to: {backup_csv_path}")

class claude:
    def __init__(self, model_name="claude-haiku-4-5-20251001"):
        """Initializes the Anthropic client and loads the API key."""
        load_dotenv(override=True)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in .env file!")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_name = model_name
        
        # Define prompts at the class level
        self.system_prompt = "Du bist ein präziser KI-Assistent für das österreichische Steuerrecht."
        # PROMPT 1: CITATION MODE
        self.citation_template = """Analysiere den folgenden steuerrechtlichen Sachverhalt. 
CRITICAL INSTRUCTION: You must respond ONLY with a semicolon-separated list of the relevant Austrian legal paragraphs. 
Always include the law book (e.g., EStG 1988) before the paragraph. Do NOT write any conversational text.

Sachverhalt:
{}

### Response:"""

        # PROMPT 2: FULL TEXT MODE (Same rules as Mistral!)
        self.full_text_template = """Du bist ein hochprofessioneller österreichischer Steuerberater.
Analysiere den folgenden steuerrechtlichen Sachverhalt.

Deine Aufgabe:
Beantworte die Frage des Nutzers fachlich korrekt, aber EXTREM KURZ UND PRÄZISE (maximal 1 bis 3 Sätze). 
Zitiere die relevanten österreichischen Paragraphen AUSSCHLIESSLICH am Ende des Satzes in Klammern (z.B. "(§ 4 Abs 1 KStG 1988)").
WICHTIG: Erstelle KEINE Listen, keine Aufzählungspunkte und schreibe nicht "Quellen:" oder "Zitierungen:". Schreibe nur fließenden Text.

Frage des Nutzers:
{}

### Antwort:"""

    def generate_responses(self, input_csv_path, backup_csv_path, limit=None, mode="citation"):
        """Reads dataset, calls Claude API, and saves line-by-line."""
        print(f"Loading data from {input_csv_path}...")
        df = pl.read_csv(input_csv_path)
        
        if limit:
            df = df.head(limit)
            
        start_index = 0

        # Define CSV headers based on mode
        if mode == "citation":
            csv_headers = ["id", "instruction", "input", "ground_truth_label", "raw_model_response"]
            max_tokens = 64
            active_template = self.citation_template
        elif mode == "full_text":
            csv_headers = ["id", "answer"]
            max_tokens = 300
            active_template = self.full_text_template
        else:
            raise ValueError("Mode must be 'citation' or 'full_text'")
        
        if os.path.exists(backup_csv_path):
            existing_df = pl.read_csv(backup_csv_path)
            start_index = existing_df.height
            print(f"Found existing backup. Resuming from case {start_index}...")
            if start_index >= df.height:
                print("Claude baseline already complete!")
                return
        else:
            print("Starting fresh Claude baseline generation...")
            with open(backup_csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "instruction", "input", "ground_truth_label", "raw_model_response"])

        #safe_client = anthropic.Anthropic(api_key=self.api_key)
        
        for i in tqdm(range(start_index, df.height), desc="Calling Claude API"):
            row = df.row(i, named=True)

            # Handle different column names between your dataset and the project's dataset
            user_text = row.get("input", row.get("prompt", ""))
            formatted_prompt = active_template.format(user_text)
            
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=0.0, # 0.0 forces the model to be as deterministic as possible
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ]
                )
                
                final_answer = response.content[0].text.strip()
                
                with open(backup_csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if mode == "citation":
                            writer.writerow([row.get("id", ""), row.get("instruction", ""), user_text, row.get("output", ""), final_answer])
                    elif mode == "full_text":
                        writer.writerow([row.get("id", ""), final_answer])                    
                # Sleep briefly to avoid API rate limits on new accounts
                time.sleep(0.2)
                
            except Exception as e:
                print(f"\nAPI Error on row {i}: {e}")
                print("Stopping loop to preserve data. You can restart the script to resume.")
                break

        print(f"\n Claude baseline complete! Results saved to: {backup_csv_path}")

class roberta:
    def __init__(self, ssd_base_path="/mnt/windows/windows_hanka_bcthesis/llm"):
        """
        Initializes the configuration, strictly routing heavy I/O to the Windows SSD.
        """
        self.CONFIG = {
            "cache_dir": Path(ssd_base_path) / "roberta_cache",
            "output_dir": Path(ssd_base_path) / "roberta_output",
            "base_model": "Stern5497/sbert-legal-xlm-roberta-base", 
            "sup_epochs": 1,     # 1 epoch is usually enough for 5,000 cases
            "batch_size": 4,     # CRITICAL: Keeps your RTX 3080 from crashing
            "warmup_steps": 100,
        }

        # # Available base models for experimentation
        # BASE_MODELS = {
        #     "mpnet": "sentence-transformers/all-mpnet-base-v2",
        #     "legal-bert": "nlpaueb/legal-bert-base-uncased",
        #     "legal-sbert": "Stern5497/sbert-legal-xlm-roberta-base",
        # }

        # Create necessary directories
        self.CONFIG["cache_dir"].mkdir(exist_ok=True)
        self.CONFIG["output_dir"].mkdir(exist_ok=True)

        print("Setup complete. Caching and Output mapped to SSD: {ssd_base_path}")

    @staticmethod
    def parse_citations(citations_raw: str) -> List[str]:
        """Simple, fast parser for clean, semicolon-separated citations."""
        if not citations_raw or citations_raw == "None":
            return []
            
        # Split by semicolon, strip whitespace, and ignore empty strings
        return [c.strip() for c in str(citations_raw).split(";") if c.strip()]

    def load_datasets(self, dataset_paths: List[str]):
        """
        Loads multiple CSVs from the provided paths, concatenates them into one 
        Polars DataFrame, and formats them for training.
        """

        dataframes = []
        for path in dataset_paths:
            print(f"Reading {path}...")
            df = pl.read_csv(path, separator=",") 
            dataframes.append(df)

        # Merge all datasets vertically into one massive dataframe
        combined_df = pl.concat(dataframes, how="vertical")

        # Validate required columns
        if "input" not in combined_df.columns or "output" not in combined_df.columns:
            raise ValueError("All datasets MUST contain 'input' and 'output' columns!")
        
        # Apply the citation parsing logic using Polars map_elements
        combined_df = combined_df.with_columns(
            pl.col("output").map_elements(self.parse_citations, return_dtype=pl.List(pl.Utf8)).alias("parsed_citations")
        )
        
        print(f"Successfully loaded and merged {len(dataset_paths)} datasets.")
        print(f"Total training cases available: {combined_df.height}")
        
        return combined_df

    def load_model(self, model_path=None):
            """Loads a model from HuggingFace OR a local SSD path."""
            # If no path is provided, use the default internet base model
            if model_path is None:
                model_path = self.CONFIG["base_model"]
                
            print(f"Loading model: {model_path}...")
            model = SentenceTransformer(model_path)
            
            # CRITICAL VRAM FIX
            model.max_seq_length = 512 
            
            return model

    #def pretrain_model() will be skipped for now as we don't need it

    def finetune_model(self, model, combined_df):
        """Supervised fine-tuning using Polars data and MNRL loss."""
        print("\n=== Stage 2: Supervised Fine-tuning ===")
        
        train_examples = []
        skipped = 0
        
        # POLARS FIX: Use iter_rows(named=True) to efficiently loop through data
        for row in combined_df.iter_rows(named=True):
            # Your case text is in the 'input' column
            case_text = row.get("input", "") 
            # We use the clean list we generated earlier
            citations = row.get("parsed_citations", []) 
            
            if not citations or not case_text:
                skipped += 1
                continue
            
            # Create a training pair for EVERY correct citation
            for citation in citations:
                if len(citation) > 3:
                    # Label=1.0 tells the model: "These two texts belong together!"
                    train_examples.append(
                        InputExample(texts=[str(case_text), str(citation)], label=1.0)
                    )
        
        print(f"Created {len(train_examples)} training pairs (skipped {skipped} rows missing data)")
        
        if len(train_examples) == 0:
            print(" WARNING: No training examples created! Check your column names.")
            return model
        
        # The DataLoader groups your pairs into batches of 4 (set in your CONFIG)
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=self.CONFIG["batch_size"]
        )
        
        # MultipleNegativesRankingLoss (MNRL) is the magic here.
        # It treats the specific citation as the "positive" match, and uses all the 
        # other citations currently in the batch of 4 as "hard negatives".
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        print("Starting training loop...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.CONFIG["sup_epochs"],
            warmup_steps=self.CONFIG["warmup_steps"],
            show_progress_bar=True
        )
        
        # Save securely to your Windows SSD
        model_path = self.CONFIG["output_dir"] / "finetuned_roberta_austrian_law"
        model.save(str(model_path))
        print(f"\n Saved fine-tuned model to: {model_path}")
        
        return model

class roberta_legal_retriever:
    def __init__(self, model, unique_citations: List[str]):
        """
        Initializes the retriever using only the unique citation strings.
        """
        self.model = model
        self.unique_citations = unique_citations
        self.index = self._build_index()

    def _build_index(self):
        """Encodes the citation strings and builds the FAISS index."""
        print("\n=== Building FAISS Index (Citation Strings Only) ===")
        print(f"Encoding {len(self.unique_citations)} unique citations...")
        
        start_time = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Encode the short citation strings
        citation_embeddings = self.model.encode(
            self.unique_citations,
            batch_size=32, # We can use a bigger batch here because the strings are very short
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device
        )
        
        print(f"Encoding took {time.time() - start_time:.2f} seconds")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(citation_embeddings.shape[1])
        index.add(citation_embeddings)
        
        print(f"Index built with {index.ntotal} vectors.")
        return index
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Takes a case text, searches FAISS, and returns the top k citations."""
        # Truncate the query to 512 tokens to match our training and save VRAM
        qvec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Search FAISS
        distances, indices = self.index.search(qvec, k=k)
        
        results = []
        for j, idx in enumerate(indices[0]):
            results.append({
                "citation": self.unique_citations[idx],
                "score": float(distances[0][j]),
                "rank": j + 1
            })
            
        return results
    
    def generate_roberta_predictions(self, test_df, output_csv_path, k=5):
        """
        Runs the retriever over a Polars test dataset and saves the output 
        in the exact semicolon-separated format needed for evaluation.
        """
        print(f"\nGenerating predictions -> {output_csv_path}")
        
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "instruction", "input", "ground_truth_label", "raw_model_response"])
            
            for i, row in enumerate(tqdm(test_df.iter_rows(named=True), total=test_df.height)):
                case_id = row.get("id", str(i))
                instruction = row.get("instruction", "Retrieve relevant Austrian laws.")
                query = row.get("input", "")
                ground_truth = row.get("output", "")
                
                # Retrieve Top-K citations using its own method
                results = self.retrieve(query, k=k)
                
                # Extract citation strings and join with semicolons
                pred_citations = [r["citation"] for r in results]
                raw_response = "; ".join(pred_citations)
                
                writer.writerow([case_id, instruction, query, ground_truth, raw_response])

class legal_rag_agent:
    def __init__(self, datasets_paths: List, roberta_ft_model_path: str):
        """
        Initializes the agent with file paths, but does NOT load the heavy models yet.
        """
        self.datasets_paths = datasets_paths
        self.roberta_ft_model_path = roberta_ft_model_path
        
        # These will hold our loaded models
        self.retriever = None
        self.mistral_model = None
        self.mistral_tokenizer = None
        
        # ==========================================
        # PROMPT 1: CITATION MODE (For your old evaluation)
        # ==========================================
        self.citation_prompt = """Du bist ein hochprofessioneller österreichischer Rechtsassistent.
Ein Retrieval-System hat den Sachverhalt analysiert und die exakten relevanten Gesetzesstellen gefunden.

Deine Aufgabe:
1. Liste die bereitgestellten Paragraphen als übersichtliche Aufzählungspunkte auf.
2. WICHTIG: Erkläre den Inhalt der Paragraphen NICHT. Erfinde keinen Text. Nenne NUR die Namen der Paragraphen.

Gefundene relevante Paragraphen:
{citations}

Sachverhalt des Nutzers:
{query}

### Antwort:
"""

        # ==========================================
        # PROMPT 2: FULL TEXT MODE (For your Professor!)
        # ==========================================
        self.full_text_prompt = """Du bist ein hochprofessioneller österreichischer Steuerberater.
Ein System hat die folgenden relevanten österreichischen Gesetzesstellen für die Frage des Nutzers gefunden:
{citations}

Deine Aufgabe:
Beantworte die Frage des Nutzers fachlich korrekt, aber EXTREM KURZ UND PRÄZISE (maximal 1 bis 3 Sätze). 
Zitiere die relevanten Paragraphen AUSSCHLIESSLICH am Ende des Satzes in Klammern (z.B. "(§ 4 Abs 1 EStG 1988)").
WICHTIG: Erstelle KEINE Listen, keine Aufzählungspunkte und schreibe nicht "Quellen:" oder "Zitierungen:". Schreibe nur fließenden Text.

Frage des Nutzers:
{query}

### Antwort:
"""

    def prepare_agent(self):
        """
        Loads RoBERTa, builds the FAISS index, and loads Mistral into VRAM. 
        You only ever need to run this ONCE per session.
        """
        print(" PREPARING RAG AGENT (This will take a minute...)")
        
        # 1. SETUP ROBERTA (The Librarian)
        print(" 1/2 Loading RoBERTa Retriever...")
        roberta_bot = roberta()
        full_df = roberta_bot.load_datasets(self.datasets_paths)
        unique_cits = (
            full_df.select(pl.col("parsed_citations").list.explode())
            .drop_nulls().unique().to_series().to_list()
        )
        
        finetuned_model = roberta_bot.load_model(self.roberta_ft_model_path)
        finetuned_model.max_seq_length = 512
        self.retriever = roberta_legal_retriever(model=finetuned_model, unique_citations=unique_cits)

        # 2. SETUP MISTRAL (The Spokesperson)
        print("\n 2/2 Loading Mistral-7B Generator...")
        mistral_bot = mistral()
        self.mistral_model, self.mistral_tokenizer = mistral_bot.load_model_and_tokenizer(
            max_seq_length=2048, 
            dtype=None, 
            load_in_4bit=True, 
            model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
        )
        
        print("\n RAG AGENT IS FULLY LOADED AND READY FOR QUERIES!")

    def ask(self, user_query: str, k: int = 3, max_tokens: int = 250, mode: str = "full_text"):
        """
        Processes the user query using the pre-loaded models.
        Use mode='citation' for old evaluation, mode='full_text' for official project output.
        """
        if self.retriever is None or self.mistral_model is None:
            raise ValueError("Models are not loaded! Please call .prepare_agent() first.")
            
        print("\n" + "="*50)
        print(" RAG AGENT PROCESSING...")
        print("="*50)
        
        # Step A: Librarian retrieves citations
        print(" 1. RoBERTa is searching the FAISS Index- searching Austrian Law Database...")
        results = self.retriever.retrieve(user_query, k=k)
        retrieved_citations = [r["citation"] for r in results]
        citations_string = "\n".join([f"- {cit}" for cit in retrieved_citations])
        
        print(f"   ↳ FOUND_retrieved_citations: {retrieved_citations}")
        #print(f"   -> FOUND_citations_string: {citations_string}")

        # 2. THE SWITCH: Choose the prompt based on the mode!
        if mode == "citation":
            active_prompt = self.citation_prompt
        elif mode == "full_text":
            active_prompt = self.full_text_prompt
        else:
            raise ValueError("Mode must be 'citation' or 'full_text'")
        
        # Step B: Format the prompt for Mistral
        print(" 2. Mistral is formulating the response...")
        prompt = active_prompt.format(citations=citations_string, query=user_query)
        inputs = self.mistral_tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Step C: Mistral generates the answer
        outputs = self.mistral_model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            use_cache=True,
            pad_token_id=self.mistral_tokenizer.eos_token_id 
        )
        
        # Step D: Decode and print
        response = self.mistral_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_answer = response.split("### Antwort:\n")[-1].strip()
        
        print("\n AGENT RESPONSE:")
        print("-" * 50)
        print(final_answer)
        print("-" * 50)
        
        return final_answer

class evaluator:
    def __init__(self, results_csv_path=None, results_df=None):
        if results_csv_path is None and results_df is None:
            raise ValueError("Either results_csv_path or results_df must be provided.")
        
        # if results_csv_path is not None:
        #     self.df = pl.read_csv(results_csv_path)
        # else:
        #     self.df = results_df

    def parse_citations(self, text):
        """Splits by semicolon, strips whitespace, and removes empty strings."""
        if not text or text == "None" or text == "":
            return []
        # Split by ;, strip spaces, and filter out empty strings
        return [c.strip() for c in str(text).split(';') if c.strip()]

    def evaluate_results(self, output_path, results_csv_path=None, results_df=None):
        if results_df is not None:
            df = results_df
        else:
            print(f"Loading results from {results_csv_path}...")
            df = pl.read_csv(results_csv_path)

        # 1. Parse strings into lists of citations
        df = df.with_columns([
            pl.col("ground_truth_label").map_elements(self.parse_citations, return_dtype=pl.List(pl.Utf8)).alias("true_list"),
            pl.col("raw_model_response").map_elements(self.parse_citations, return_dtype=pl.List(pl.Utf8)).alias("pred_list")
        ])

        # 2. Calculate Set Metrics row by row
        # We use basic Python sets inside map_elements for reliable comparison
        def calculate_metrics(row):
            true_set = set(row[0])
            pred_set = set(row[1])
            
            # Intersection: What did the model get right?
            correct_predictions = len(true_set.intersection(pred_set))
            
            # Exact Match Boolean
            exact_match = (true_set == pred_set)
            
            # Precision: Correct / Total Predicted aka "how many citations did he got correct / how many citations did he overall predicted?"
            precision = correct_predictions / len(pred_set) if len(pred_set) > 0 else 0.0
            
            # Recall: Correct / Total True aka "how many citations did he predict correctly / how many correct citations are originally there?"
            recall = correct_predictions / len(true_set) if len(true_set) > 0 else 0.0
            
            return {"exact_match": exact_match, "precision": precision, "recall": recall}

        # Apply the calculation
        metrics_df = df.select(["true_list", "pred_list"]).map_rows(
            lambda row: calculate_metrics(row),
            return_dtype=pl.Struct([
                pl.Field("exact_match", pl.Boolean),
                pl.Field("precision", pl.Float64),
                pl.Field("recall", pl.Float64)
            ])
        ).unnest("map")

        # Merge metrics back into the main dataframe
        df = pl.concat([df, metrics_df], how="horizontal")

        # # Calculate F1 Score per row: 2 * (Precision * Recall) / (Precision + Recall)
        # df = df.with_columns(
        #     pl.when((pl.col("precision") + pl.col("recall")) > 0)
        #     .then(2 * (pl.col("precision") * pl.col("recall")) / (pl.col("precision") + pl.col("recall")))
        #     .otherwise(0.0)
        #     .alias("f1_score")
        # )

        print(df)

        # 3. Aggregate Final Results
        total_cases = df.height
        exact_match_acc = df["exact_match"].sum() / total_cases
        avg_precision = df["precision"].mean()
        avg_recall = df["recall"].mean()
        #avg_f1 = df["f1_score"].mean()

        print("\n" + "="*40)
        print("BASELINE EVALUATION RESULTS")
        print("="*40)
        print(f"Total Cases Evaluated: {total_cases}")
        print(f"Exact Match Accuracy:  {exact_match_acc:.2%}")
        print(f"Average Precision:     {avg_precision:.2%}")
        print(f"Average Recall:        {avg_recall:.2%}")
        #print(f"Average F1-Score:      {avg_f1:.2%}")
        print("="*40)

        df_flattened = df.select([
            pl.col("id"),
            pl.col("instruction"),
            pl.col("input"),
            pl.col("ground_truth_label"),
            pl.col("raw_model_response"),
            pl.col("exact_match"),
            pl.col("precision"),
            pl.col("recall")
        ])
        
        if output_path is not None:
            # Optional: Save the evaluated dataframe so you can inspect where it failed
            df_flattened.write_csv(output_path)
            print(f"Detailed row-by-row metrics saved to: {output_path}")

