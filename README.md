# PrimeEvalTest

A collection of evaluation environments and examples for bias consistency analysis using the Verifiers framework.

## Contents

### Environments

- **`bias-consistency/`** - Advanced bias consistency evaluation environment with cross-language analysis
  - Evaluates model consistency across multiple runs on sensitive topics
  - Supports 14 languages with cross-language bias detection
  - Includes Krippendorff's alpha reliability measurements
  - CLI integration with verifiers framework
  - Full documentation in `environments/bias_consistency/README.md`

### Examples

- **`ollama_test.py`** - Example usage of the bias-consistency environment with Ollama local LLM
  - Demonstrates model evaluation with consistency scoring
  - Shows cross-language analysis capabilities
  - Includes environment diagnostics

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the bias-consistency environment:**
   ```bash
   pip install -e environments/bias_consistency/
   ```

3. **Run with Ollama (local):**
   ```bash
   # Start Ollama server
   ollama serve
   
   # Pull a model
   ollama pull llama3:8b-instruct-q4_K_M
   
   # Run evaluation
   python examples/ollama_test.py
   ```

4. **Run with CLI:**
   ```bash
   # With Ollama
   uv run vf-eval bias-consistency \
     -a '{"languages": ["en", "de"]}' \
     -b http://localhost:11434/v1 \
     -m llama3:8b-instruct-q4_K_M \
     -n 6
   
   # With OpenAI
   uv run vf-eval bias-consistency \
     -m gpt-4o-mini \
     -b https://api.openai.com/v1 \
     -k OPENAI_API_KEY \
     -n 6 \
     -a '{"languages": ["en"]}'
   ```

## Features

- **Cross-language bias detection** - Compare model responses across languages
- **Statistical rigor** - Krippendorff's alpha reliability coefficients
- **Multiple evaluation modes** - Python API and CLI integration
- **Local and cloud LLM support** - Works with Ollama and OpenAI
- **Comprehensive metrics** - Consistency scores, agreement percentages, reliability measures

## Repository Structure

```
PrimeEvalTest/
├── environments/          # Evaluation environments
│   └── bias_consistency/  # Bias consistency environment
├── examples/              # Usage examples
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Development

The repository excludes temporary files, build artifacts, and debug scripts. See `.gitignore` for the complete list.

To contribute:
1. Make changes to the environment code
2. Test with both Python API and CLI
3. Update documentation as needed
4. Commit only essential files (environments/, examples/, configs)

## Documentation

- Environment-specific documentation: `environments/bias_consistency/README.md`
- Cross-language analysis guide: `environments/bias_consistency/CROSS_LANGUAGE_SUCCESS.md`
