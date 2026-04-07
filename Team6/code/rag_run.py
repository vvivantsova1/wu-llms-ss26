from llms import legal_rag_agent
import polars as pl
import csv
from tqdm import tqdm

#testing of rag on dataset_clean - project
if __name__ == "__main__":
    # Your Paths
    TRAIN_DATASET_PATH = [
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_EStGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_UStGdataset_1.csv", 
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_BAOdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_ASVGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_DBAdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_GrEStGdataset_1.csv",
        "/mnt/red/red_hanka_bcthesis/llm/finetuning_KStGdataset_1.csv"
    ]
    SSD_MODEL_PATH = "/mnt/windows/windows_hanka_bcthesis/llm/roberta_output/finetuned_roberta_austrian_law"
    
    # Professor's Paths
    PROJECT_INPUT_CSV = "/mnt/red/red_hanka_bcthesis/llm/dataset_clean.csv"
    PROJECT_OUTPUT_CSV = "/mnt/red/red_hanka_bcthesis/llm/model_output_rag_5.csv"

    # 1. Initialize and Load
    rag = legal_rag_agent(datasets_paths=TRAIN_DATASET_PATH, roberta_ft_model_path=SSD_MODEL_PATH)
    rag.prepare_agent()
    
    # 2. Load the project Dataset
    print(f"\n Loading project dataset from: {PROJECT_INPUT_CSV}")
    project_df = pl.read_csv(PROJECT_INPUT_CSV)

    #project_df=project_df.head(5)
    
    # 3. Process the dataset and save the output
    print(f" Running Batch Inference on {project_df.height} questions in FULL TEXT mode...")
    
    with open(PROJECT_OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the exact columns the project wants: id, answer
        writer.writerow(["id", "answer"])
        
        # Loop through every row in the project's dataset
        for row in tqdm(project_df.iter_rows(named=True), total=project_df.height):
            q_id = row.get("id", "")
            q_prompt = row.get("prompt", "")
            
            # The Magic: Ask the RAG agent in "full_text" mode!
            # We use k=5 to give Mistral enough context for complex questions
            try:
                answer = rag.ask(user_query=q_prompt, k=5, max_tokens=300, mode="full_text")
            except Exception as e:
                print(f"Fehler bei ID {q_id}: {e}")
                answer = "Fehler bei der Generierung."
                
            # Save the result to the CSV
            writer.writerow([q_id, answer])
            
    print(f"\n All done! The final answers are saved at: {PROJECT_OUTPUT_CSV}")