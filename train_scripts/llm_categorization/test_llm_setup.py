"""
Test LLM Integration Setup

This script tests if the LLM integration module can be imported and works correctly.
Run this before running the actual training scripts.
"""

import sys
import os
from pathlib import Path

def test_llm_integration():
    """Test if LLM integration works"""
    print("🧪 Testing LLM Integration Setup...")
    
    # Add path to LLM categorization
    llm_path = Path(__file__).parent.parent / "llm_categorization"
    sys.path.append(str(llm_path))
    
    try:
        # Test import
        print("📦 Testing imports...")
        from llm_integration import create_llm_enhancer, get_llm_model_config
        print("✅ Imports successful!")
        
        # Test enhancer creation
        print("🔧 Testing enhancer creation...")
        enhancer = create_llm_enhancer(enable_llm=False, sample_limit=10)  # Disabled for testing
        print("✅ Enhancer created successfully!")
        
        # Test model config
        print("⚙️ Testing model configuration...")
        config = get_llm_model_config(
            base_output_dir="test_output",
            base_wandb_name="test_run",
            enable_llm=False
        )
        print(f"✅ Model config: {config}")
        
        return True, "All tests passed!"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

if __name__ == "__main__":
    success, message = test_llm_integration()
    
    if success:
        print(f"\n🎉 {message}")
        print("🚀 You can now run the training scripts!")
    else:
        print(f"\n❌ Test failed: {message}")
        print("🔧 Please fix the issues before running training scripts.")
        
    print(f"\n📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python path includes: {sys.path[-3:]}")  # Show last 3 paths