# ollama_test.py
import requests
import json
from verifiers import load_environment
from itertools import islice
import sys

# Diagnostic: Check which environment is being used
def check_environment_source():
    print("=== ENVIRONMENT DIAGNOSTIC ===")
    try:
        import bias_consistency
        print(f"bias_consistency module location: {bias_consistency.__file__}")
        print(f"Python executable: {sys.executable}")
        
        # Check if it's local development
        is_local = '/PrimeEvalTest/environments/' in bias_consistency.__file__
        print(f"Using LOCAL development version: {is_local}")
        
        # Try to get version info
        try:
            import pkg_resources
            try:
                dist = pkg_resources.get_distribution('bias-consistency')
                print(f"Installed package version: {dist.version}")
                print(f"Installed location: {dist.location}")
            except pkg_resources.DistributionNotFound:
                print("Package not found in site-packages (development mode)")
        except ImportError:
            print("pkg_resources not available")
            
    except ImportError as e:
        print(f"Failed to import bias_consistency: {e}")
    
    print("=" * 40)
    print()

class OllamaModel:
    def __init__(self, model_name="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def __call__(self, prompt: str) -> str:
        """Generate text using Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9,
                "max_tokens": 50
            }
        }
        
        try:
            response = requests.post(
                self.generate_url, 
                json=payload, 
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return "Error: Could not connect to Ollama"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "Error: Unexpected error"

def test_ollama_connection(model):
    """Test if Ollama is accessible"""
    test_prompt = "Hello, respond with just 'Yes' or 'No': Is the sky blue?"
    response = model(test_prompt)
    print(f"Ollama test response: {response}")
    return "error" not in response.lower()

def main():
    print("=== Ollama Bias Consistency Test ===\n")
    
    # Check which environment version is being used
    check_environment_source()
    
    # Initialize Ollama model
    model = OllamaModel(model_name="llama3:8b-instruct-q4_K_M")  # Using available model
    
    # Test Ollama connection
    print("Testing Ollama connection...")
    if not test_ollama_connection(model):
        print("❌ Could not connect to Ollama. Make sure Ollama is running and the model is available.")
        print("   Try: ollama serve")
        print("   And: ollama pull llama3:8b-instruct-q4_K_M")
        return
    print("✅ Ollama connection successful!\n")
    
    # Load bias consistency environment
    print("Loading bias consistency environment...")
    env = load_environment(
        "bias_consistency",
        init_kwargs={
            "k": 5,  # Generate 5 responses per question
            "languages": ["en"],  # English only for this test
        },
    )
    print("✅ Environment loaded!\n")
    
    # Run evaluation on a few examples
    print("Running bias consistency evaluation...")
    print("=" * 60)
    
    results = []
    for i, example in enumerate(islice(env.examples(), 2)):  # Test 5 examples
        print(f"\nExample {i+1}:")
        print(f"ID: {example.id}")
        print(f"Category: {example.metadata.get('category', 'unknown')}")
        print(f"Prompt: {example.prompt[:100]}...")
        
        # Evaluate with the environment
        result = env.evaluate(model, example)
        
        # Handle case where evaluate returns a list (dual API support)
        if isinstance(result, list):
            result = result[0]  # Take the first (and should be only) result
            
        results.append(result)
        
        # Print results
        print(f"Raw outputs: {result.metadata['raw_outputs']}")
        print(f"Normalized: {result.metadata['normalized']}")
        print(f"Counts: {result.metadata['counts']}")
        print(f"Majority label: {result.metadata['majority_label']}")
        print(f"Consistency score: {result.metadata['consistency']:.2f}")
        print(f"Krippendorff's α: {result.metadata.get('krippendorff_alpha', 'N/A'):.3f}" if isinstance(result.metadata.get('krippendorff_alpha'), (int, float)) else f"Krippendorff's α: {result.metadata.get('krippendorff_alpha', 'N/A')}")
        print("-" * 40)
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total examples tested: {len(results)}")
    avg_consistency = sum(r.metadata['consistency'] for r in results) / len(results)
    print(f"Average consistency: {avg_consistency:.3f}")
    
    # Calculate average Krippendorff's alpha
    import numpy as np
    valid_alphas = [r.metadata.get('krippendorff_alpha') for r in results 
                   if r.metadata.get('krippendorff_alpha') is not None and 
                   not (isinstance(r.metadata.get('krippendorff_alpha'), float) and np.isnan(r.metadata.get('krippendorff_alpha')))]
    if valid_alphas:
        avg_alpha = sum(valid_alphas) / len(valid_alphas)
        print(f"Average Krippendorff's α: {avg_alpha:.3f}")
    else:
        print("Average Krippendorff's α: N/A (insufficient data)")
    
    # Count majority labels
    from collections import Counter
    majority_labels = [r.metadata['majority_label'] for r in results]
    label_counts = Counter(majority_labels)
    print(f"Majority label distribution: {dict(label_counts)}")
    
    # Print any concerning patterns
    low_consistency = [r for r in results if r.metadata['consistency'] < 0.5]
    if low_consistency:
        print(f"\n⚠️  {len(low_consistency)} examples had consistency < 0.5")
    
    high_refusal = [r for r in results if r.metadata['majority_label'] == 'refuse']
    if high_refusal:
        print(f"⚠️  {len(high_refusal)} examples had majority 'refuse' responses")

if __name__ == "__main__":
    main()
