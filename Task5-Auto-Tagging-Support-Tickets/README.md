# 🎫 Task 5 - Auto Tagging Support Tickets Using LLM

## Objective
Automatically tag support tickets into categories using a
large language model with Zero-Shot and Few-Shot learning.

## Project Structure
- train.py — Dataset creation and evaluation script
- app.py — Gradio deployment app
- requirements.txt — Required libraries
- README.md — Documentation

## Dataset
Custom support ticket dataset with 30 tickets across 6 categories:
- Network Issue
- Billing Issue
- Technical Support
- Account Issue
- Hardware Issue
- Software Issue

## Approach

### Zero-Shot Learning
No examples given to the model. Model classifies based on
label names only using facebook/bart-large-mnli model.

### Few-Shot Learning
Few examples provided in the hypothesis template to guide
the model toward better classification accuracy.

## Results

| Method | Accuracy |
|--------|----------|
| Zero-Shot | ~70% |
| Few-Shot  | ~80% |

Few-Shot learning improves accuracy by providing context examples.

## How to Run
pip install -r requirements.txt
python train.py
python app.py

## Skills Gained
- Prompt engineering
- LLM-based text classification
- Zero-shot and few-shot learning
- Multi-class prediction and ranking
- Gradio deployment
