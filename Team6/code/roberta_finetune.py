from llms import roberta, roberta_legal_retriever
import polars as pl

if __name__ == "__main__":
    # 1. Initialize the Finetuner
    roberta_bot = roberta()

    # 2. Load your datasets
    cases_datasets_paths = [
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_EStGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_UStGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_BAOdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_ASVGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_DBAdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_GrEStGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_KStGdataset_1.csv"
    ]
    train_df = roberta_bot.load_datasets(cases_datasets_paths)

    # # 🔥 THE SMOKE TEST HACK 🔥
    # # This chops your massive dataset down to just the first 10 rows for testing
    # train_df = train_df.head(10)
    # print(f"⚠️ SMOKE TEST MODE: Only training on {train_df.height} cases.")

    # 3. Load Base Model and Fine-Tune
    base_model = roberta_bot.load_model()
    finetuned_roberta = roberta_bot.finetune_model(base_model, train_df)

    # 4. Extract unique citations for the FAISS Hack
    print("\nExtracting unique citations for FAISS index...")
    # Polars magic to get a flat list of unique citations
    unique_cits = (
        train_df.select(pl.col("parsed_citations").list.explode())
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )

    # 5. Build the Retriever (The Librarian)
    retriever = roberta_legal_retriever(model=finetuned_roberta, unique_citations=unique_cits)

    # 6. Run a Quick Sanity Check Test
    print("\n" + "="*50)
    print("LIVE RETRIEVAL TEST")
    print("="*50)

    # Testen wir direkt das KStG (Körperschaftsteuer), um zu sehen, ob es gelernt hat!
    test_case = "Welche steuerlichen Konsequenzen hat es, wenn eine GmbH ein zinsloses Darlehen an ihren Gesellschafter vergibt?"
    print(f"QUERY: {test_case}\n")

    results = retriever.retrieve(test_case, k=5)
    for r in results:
        print(f"Rank {r['rank']}: {r['citation']} (Score: {r['score']:.4f})")