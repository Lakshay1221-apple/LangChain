# LangChain Content Studio

A guided Jupyter Notebook that uses LangChain and Google Gemini to transform an article into polished supporting content.

## What It Demonstrates

- LCEL prompt and model composition
- Separate deterministic and creative Gemini model configurations
- Article title generation
- SEO-friendly description generation
- Structured paragraph editing with Pydantic
- Generation of a text prompt for an external image-generation model

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or VS Code with the Jupyter extension
- A Google Gemini API key

The notebook installs these packages when its first code cell runs:

- `langchain-core`
- `langchain-google-genai`
- `langchain-community`

## Setup

1. Clone or open this repository.
2. Open `langchain_1.ipynb` in Jupyter or VS Code.
3. Set the API key before running the notebook:

   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```

   Alternatively, leave the variable unset and enter the key when the notebook prompts for it. Do not commit API keys to the repository.

4. Run the cells from top to bottom.

## Workflow

The notebook follows a simple content pipeline:

1. Install the LangChain dependencies.
2. Configure two `ChatGoogleGenerativeAI` clients:
   - `llm` uses temperature `0.0` for consistent output.
   - `creative_llm` uses temperature `0.7` for more varied titles and edits.
3. Define the source article in `article`.
4. Generate an article title with `chain_one`.
5. Generate an SEO description with `chain_two`.
6. Review and improve a paragraph with `chain_three`, which returns structured `Paragraph` output.
7. Generate an image description with `chain_four`.

## Outputs

The main results are stored in these variables:

- `article_title_msg["article_title"]`
- `article_description_msg["summary"]`
- `out["original_paragraph"]`
- `out["edited_paragraph"]`
- `out["feedback"]`
- `generated_image_description`

## Notes

The final step creates a textual image prompt only. It does not generate an image. To create an image, send `generated_image_description` to a compatible image-generation service.

Model availability and API behavior can change over time. If `gemini-2.5-flash` is unavailable for your account, select a currently supported Gemini model in the notebook.

## Project Structure

```text
.
├── langchain_1.ipynb  # Guided LangChain and Gemini workflow
└── README.md          # Project documentation
```
