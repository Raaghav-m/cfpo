# CFPO Process Diagram

## Mermaid Code

```mermaid
flowchart LR
    IP[Input Prompt] --> A

    A[Dataset & Task Preparation<br/>• Optimization Set<br/>• Validation Set<br/>• Initial Prompt] --> B

    B[Structured Prompt<br/>Decomposition<br/>• Decompose Prompt<br/>• Content & Format Setup] --> C

    C[Content & Format<br/>Optimization<br/>• Improve Instructions<br/>• Refine Examples<br/>• Test Prompt Formats<br/>• Select Best Layout] --> D

    D[LLM Evaluation<br/>• Target LLM<br/>• Dataset Examples<br/>• Output Comparison]

    D -->|Feedback & Results| B
    C -->|Updated Prompt| A
```

## Description of Changes Made

1. **Removed all colors** - The diagram uses default black/white styling (no fill colors)
2. **Added Input Prompt arrow** - New "Input Prompt" node connects to "Dataset & Task Preparation"
3. **Renamed block** - "Structured Prompt Initialization" → "Structured Prompt Decomposition"
4. **Removed Validation & Testing block** - Replaced with "LLM Evaluation" in its position
5. **Merged optimization blocks** - Combined "Content Optimization" and "Format Optimization" into single "Content & Format Optimization" block

## How to View

1. Copy the mermaid code above
2. Paste into [Mermaid Live Editor](https://mermaid.live/)
3. Export as PNG/SVG

Or open the `cfpo_diagram.html` file in a browser to see the rendered diagram.
