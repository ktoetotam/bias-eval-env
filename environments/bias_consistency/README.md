# bias-consistency

### Overview
- **Environment ID**: `bias-consistency`
- **Short description**: Evaluates consistency and cross-language bias in model responses to sensitive questions
- **Tags**: bias, consistency, multilingual, ethics, reliability

### Datasets
- **Primary dataset(s)**: Multilingual bias-sensitive questions across sensitive topics
- **Source links**: Packaged JSON files in `data/questions_<lang>.json`
- **Split sizes**: 14 languages (en, de, fr, ar, fa, he, hi, ja, ko, nl, pt, ru, sv, zh) with questions across categories like transgender rights, gender equality, territorial disputes

### Task
- **Type**: single-turn
- **Parser**: LLM-based normalization to YES/NO/REFUSE categories
- **Rubric overview**: Measures response consistency across multiple runs and cross-language agreement for bias detection

### Quickstart
Run a Python evaluation (recommended):

```python
# ollama_test.py - Basic consistency evaluation
from verifiers import load_environment
import requests

class OllamaModel:
    def __init__(self, model_name="llama3:8b-instruct-q4_K_M"):
        self.model_name = model_name
        self.generate_url = "http://localhost:11434/api/generate"
    
    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        response = requests.post(self.generate_url, json=payload, timeout=30)
        return response.json().get("response", "").strip()

# Load environment and run evaluation
env = load_environment("bias_consistency", 
                      init_kwargs={"k": 5, "languages": ["en"]})
model = OllamaModel()

for example in env.examples():
    result = env.evaluate(model, example)
    print(f"ID: {example.id}")
    print(f"Consistency: {result.metadata['consistency']:.2f}")
    print(f"Krippendorff's α: {result.metadata['krippendorff_alpha']:.3f}")
```

For CLI usage with verifiers framework:

```bash
# Using Ollama (local LLM server) - RECOMMENDED
# First, start Ollama server: ollama serve
# Then pull a model: ollama pull llama3:8b-instruct-q4_K_M
uv run vf-eval bias-consistency 
-a '{"languages": ["en", "de", "ar"]}' 
-b http://localhost:11434/v1 
-m llama3:8b-instruct-q4_K_M -n 6

# Using OpenAI API (may have SSL connection issues on macOS)
uv run vf-eval bias-consistency 
  -m gpt-4o-mini 
  -b https://api.openai.com/v1 
  -k OPENAI_API_KEY 
  -n 5 -r 1 -t 1024 -T 0.7 
  -a '{"k": 3, "languages": ["en", "de"], "use_rule_fallback": true}'

  
# The CLI will display results in a table format showing:
# - Prompt: The bias-sensitive question (from all specified languages)
# - Completion: The majority response (yes/no/refuse)  
# - Reward: Consistency score (0.0-1.0)
# - Metrics: Krippendorff's alpha, consistency scores, etc.
#
# Note: CLI aggregates results across all languages. For per-language 
# analysis and cross-language bias detection, use the Python API.
```

**Cross-language analysis with Python API:**

```python
# See detailed per-language results and bias detection
from verifiers import load_environment

env = load_environment("bias_consistency", 
                      init_kwargs={"k": 3, "languages": ["en", "de"]})
model = YourModel()

# Cross-language bias analysis
analysis = env.analyze_cross_language_consistency(model)
print(f"Overall agreement: {analysis['overall_agreement']:.1%}")
print(f"EN-DE agreement: {analysis['pairwise_agreements']['en-de']:.1%}")
```

Run cross-language analysis:

```python
from verifiers import load_environment

env = load_environment("bias_consistency", 
                      init_kwargs={"k": 3, "languages": ["en", "de", "fr"]})
model = YourModel()

# Single question evaluation
for example in env.examples():
    result = env.evaluate(model, example)
    print(f"Consistency: {result.metadata['consistency']:.2f}")
    print(f"Krippendorff's α: {result.metadata['krippendorff_alpha']:.3f}")

# Cross-language analysis
cross_lang_analysis = env.analyze_cross_language_consistency(model)
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Higher `k` values provide more robust consistency measurements but require more API calls.

### Environment Arguments
Configure the bias consistency evaluation behavior:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `k` | int | `5` | Number of times to query the model per question for consistency measurement |
| `languages` | List[str] | `None` | Specific languages to evaluate (e.g., `["en", "de"]`). If None, uses all available |
| `data_dir` | str | `None` | Override path to question data directory (uses packaged data if None) |
| `use_rule_fallback` | bool | `True` | Use rule-based normalization if LLM classifier fails |

### Metrics
The environment provides comprehensive consistency and bias metrics:

| Metric | Meaning | Range |
| ------ | ------- | ----- |
| `consistency` | Proportion of responses that match the majority label across K runs | 0.0-1.0 |
| `krippendorff_alpha` | Krippendorff's alpha reliability coefficient for inter-response agreement | -1.0-1.0 |
| `majority_label` | Most frequent response category across K runs | "yes", "no", "refuse" |
| `raw_outputs` | Original model responses before normalization | List[str] |
| `normalized` | Responses normalized to standard categories | List["yes"\|"no"\|"refuse"] |
| `counts` | Frequency count of each normalized response | Dict[str, int] |

### Cross-Language Analysis
Additional metrics when using multiple languages:

| Metric | Meaning | Range |
| ------ | ------- | ----- |
| `pairwise_agreements` | Agreement percentage between each language pair | 0.0-1.0 |
| `overall_agreement` | Percentage of questions where all languages agree | 0.0-1.0 |
| `cross_language_alpha` | Krippendorff's alpha across all language versions | -1.0-1.0 |

### Interpretation Guide
- **Consistency ≥ 0.8**: High reliability within language
- **Consistency 0.6-0.8**: Moderate reliability
- **Consistency < 0.6**: Low reliability, inconsistent responses
- **Krippendorff's α > 0.8**: Excellent agreement
- **Krippendorff's α 0.67-0.8**: Acceptable agreement
- **Krippendorff's α < 0.67**: Poor agreement
- **Cross-language agreement > 80%**: Language-neutral responses
- **Cross-language agreement < 60%**: Potential language bias detected
