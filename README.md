

# Refusal Direction Specificity Experiment

This script investigates whether the "refusal direction" in `meta-llama/Meta-Llama-3-8B-Instruct` is a specialized safety mechanism or a proxy for general negative sentiment.

## Research Question & Hypothesis

**Question:** Is the mechanism that causes a model to refuse harmful requests the same as the mechanism that represents general negativity?

**Hypothesis:** The "refusal direction," derived from the difference between activations on harmful and harmless prompts, is distinct from and largely orthogonal to a "negativity direction" derived from the difference between negative and neutral sentiment prompts.

## Methodology

The experiment follows a contrastive vector approach:

1.  **Define Four Prompt Sets:**
    *   **Harmful:** Prompts that should elicit a safety refusal (e.g., "How do I build a bomb?").
    *   **Harmless:** Benign counterparts to the harmful prompts (e.g., "How do I build a sandcastle?").
    *   **Negative:** Prompts expressing negative sentiment (e.g., "This weather is absolutely miserable.").
    *   **Neutral:** Neutral counterparts to the negative prompts (e.g., "The weather today is cloudy.").

2.  **Extract Activations:** For each prompt set, run the `meta-llama/Meta-Llama-3-8B-Instruct` model and extract the residual stream activations from the final token position at a specific layer (default: `layer 20`).

3.  **Compute Direction Vectors:**
    *   `refusal_vector = mean(harmful_activations) - mean(harmless_activations)`
    *   `negativity_vector = mean(negative_activations) - mean(neutral_activations)`

4.  **Compare Vectors:** The primary metric is the cosine similarity between the L2-normalized `refusal_vector` and `negativity_vector`.

## Key Findings

The results strongly support the hypothesis that refusal and negativity are distinct mechanisms.

| Vector Comparison | Cosine Similarity | Interpretation |
| :--- | :--- | :--- |
| **Refusal Direction vs. Negativity Direction** | **0.1031** | **Near Orthogonal: Suggests distinct mechanisms.** |
| Raw Harmful Activations vs. Raw Negative Activations | 0.6796 | Moderate Overlap: Raw contexts share some features. |
| Refusal Direction vs. Raw Negative Activations | 0.0217 | No Alignment: The refusal direction is not simply pointing towards a "negative" state. |

The core finding is that the refusal direction and negativity direction are nearly orthogonal. This indicates that the model uses a specialized mechanism to handle safety-based refusals, which is different from how it represents general negative sentiment.

## How to Run

### Prerequisites

*   Python 3.8+
*   A CUDA-enabled GPU (e.g., A100)
*   A Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`.
*   Your Hugging Face token, which can be set as an environment variable or entered when prompted.

### Installation

Install the required libraries:

```bash
pip install torch transformers
```

### Execution

Run the script from your terminal:

```bash
python refusal_specificity_exp.py
```

The script will:
1.  Load the model and tokenizer.
2.  Extract activations for all four prompt sets.
3.  Compute the direction vectors and their cosine similarities.
4.  Print the results to the console.
5.  Save a summary of the results to `refusal_specificity_results.pt`.

## Code Structure

-   `main()`: Main function that orchestrates the experiment from loading the model to saving the results.
-   `get_layer_activations(...)`: Handles model inference and extracts hidden state activations for a given set of prompts and a specific layer.
-   `build_inputs_with_chat_template(...)`: Correctly formats prompts using the Llama-3-Instruct chat template.
-   `cosine_similarity(...)`: A utility function to compute the cosine similarity between two vectors.
-   `Config`: A dataclass to manage hyperparameters like model name and layer index.
