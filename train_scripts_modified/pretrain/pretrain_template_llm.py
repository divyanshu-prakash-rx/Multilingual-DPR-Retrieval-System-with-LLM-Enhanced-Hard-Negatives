"""
LLM-Enhanced Training Template

This template shows how to integrate LLM enhancement into any negative sampling method.
It can be adapted for ANCE, DPR, ICT, TAS, or any other method.

Key Features:
- Automatic LLM availability detection
- Configurable sample processing limits
- Graceful fallback to baseline training
- Consistent output paths and experiment naming
- Production-ready error handling

Usage:
    1. Set METHOD_NAME to your training method (e.g., 'ANCE', 'DPR-Base')
    2. Configure method-specific parameters
    3. Run the script
"""

import logging
import pandas as pd
import sys
import os
from pathlib import Path
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

# Add LLM categorization to path
sys.path.append(str(Path(__file__).parent.parent / "llm_categorization"))

# Import common LLM integration
try:
    from llm_integration import create_llm_enhancer, get_llm_model_config
except ImportError:
    # Alternative import path
    sys.path.append(str(Path(__file__).parent.parent))
    from llm_categorization.llm_integration import create_llm_enhancer, get_llm_model_config

# Configuration
METHOD_NAME = "TEMPLATE"  # Change this to your method (e.g., "ANCE", "DPR-Base")
ENABLE_LLM = True  # Set to False for baseline training
LLM_SAMPLE_LIMIT = 1000  # Number of samples to process with LLM

# Setting up the logging configuration
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def main():
    """Main training function"""
    
    # Path to the training data
    train_data_path = "../../data/msmarco/msmarco-train.tsv"
    
    # Load training data
    if train_data_path.endswith(".tsv"):
        train_data = pd.read_csv(train_data_path, sep="\t")
    else:
        train_data = train_data_path
    
    print(f"📊 Loaded training data: {len(train_data)} samples")
    
    # Create LLM enhancer
    enhancer = create_llm_enhancer(
        enable_llm=ENABLE_LLM,
        sample_limit=LLM_SAMPLE_LIMIT,
        verbose=True
    )
    
    # Enhance training data with LLM
    train_data = enhancer.enhance_training_data(
        train_data, 
        method=METHOD_NAME
    )
    
    # Setup model configuration paths
    base_output_dir = f"../../trained_models/pretrained/{METHOD_NAME}-msmarco"
    base_wandb_name = f"{METHOD_NAME}-msmarco"
    
    model_config = get_llm_model_config(
        base_output_dir=base_output_dir,
        base_wandb_name=base_wandb_name,
        enable_llm=ENABLE_LLM
    )
    
    # Setting up the model arguments (customize for your method)
    model_args = RetrievalArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.use_cached_eval_features = False
    model_args.include_title = False if "msmarco" in train_data_path else True
    model_args.max_seq_length = 256
    model_args.num_train_epochs = 40
    model_args.train_batch_size = 8
    model_args.use_hf_datasets = True
    model_args.learning_rate = 1e-6
    model_args.warmup_steps = 5000
    model_args.save_steps = 300000
    model_args.evaluate_during_training = False
    model_args.save_model_every_epoch = False
    
    # Enabling hard negatives for training
    model_args.hard_negatives = True
    
    # Setting up the number of GPUs to use and the data format
    model_args.n_gpu = 1
    model_args.data_format = "beir"
    
    # Method-specific configurations (customize for your method)
    if METHOD_NAME == "ANCE":
        model_args.ance_training = True
        model_args.ance_refresh_n_epochs = 10
    elif METHOD_NAME.startswith("DPR"):
        # DPR-specific settings
        pass
    elif METHOD_NAME.startswith("ICT"):
        # ICT-specific settings
        pass
    elif METHOD_NAME.startswith("TAS"):
        # TAS-specific settings
        pass
    
    # Setting up the model type, model name, context name, and question name
    model_type = "custom"
    model_name = None
    context_name = "bert-base-multilingual-cased"
    question_name = "bert-base-multilingual-cased"
    
    # Setting up the project name for Weights & Biases integration
    model_args.wandb_project = "Negative Sampling Multilingual - Pretrain"
    model_args.wandb_kwargs = model_config['wandb_kwargs']
    
    # Setting up the output directory (with LLM suffix if enhanced)
    model_args.output_dir = model_config['output_dir']
    
    print(f"💾 Model will be saved to: {model_args.output_dir}")
    print(f"🏷️ Experiment name: {model_config['wandb_kwargs']['name']}")
    print(f"🤖 LLM Enhanced: {model_config['llm_enabled']}")
    
    # Create RetrievalModel
    model = RetrievalModel(
        model_type,
        model_name,
        context_name,
        question_name,
        args=model_args,
    )
    
    # Training the model
    print(f"🚀 Starting {'LLM-Enhanced' if model_config['llm_enabled'] else 'Baseline'} {METHOD_NAME} Training")
    print(f"📊 Training samples: {len(train_data)}")
    
    model.train_model(
        train_data,
        eval_set="dev",
    )
    
    print(f"✅ Training completed! Model saved to: {model_args.output_dir}")

if __name__ == "__main__":
    # Setting the start method for multiprocessing
    from multiprocess import set_start_method
    set_start_method("spawn")
    
    main()