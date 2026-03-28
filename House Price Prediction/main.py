import time
from src.train_model import train_multiple_models
from src.evaluate import evaluate_and_compare_all

def run_end_to_end_pipeline():
    print("=" * 60)
    print("🚀 INITIALIZING END-TO-END MACHINE LEARNING PIPELINE 🚀")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Data Ingestion, Preprocessing, and Training
    print("\n[PHASE 1]: Data Processing & Model Training")
    print("-" * 40)
    # This automatically calls load_data() and preprocess_data() internally
    train_multiple_models() 
    
    # Step 2: Benchmarking and Evaluation
    print("\n[PHASE 2]: Leaderboard Benchmarking & Analytics")
    print("-" * 40)
    evaluate_and_compare_all()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 60)
    print(f"✅ PIPELINE EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 60)

if __name__ == "__main__":
    run_end_to_end_pipeline()
