# Tic-tac-toe AI Experiment

We use tic-tac-toe as a toy problem here to learn how to train an AI to play it well.

The goal is to have the AI -
1. Understand tic-tac-toe grid pictured through a phone camera
2. Offer a formidable (perfect?) gameplay as an opponent

# Running

## Generate Examples

```sh
python -m src.llm_experiments.llm_generate_data --player "o3-mini"
```

## Evaluate
Use [chart_results](./chart_results.ipynb) to evaluate and visualize.

## Demo

A demo / playground for the evaluator can be served via -

```sh
./run_ui_demo.sh
```
