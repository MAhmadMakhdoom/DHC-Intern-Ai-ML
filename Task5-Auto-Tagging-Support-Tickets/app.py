import gradio as gr
from transformers import pipeline
import torch

ALL_TAGS = [
    "Network Issue",
    "Billing Issue",
    "Technical Support",
    "Account Issue",
    "Hardware Issue",
    "Software Issue",
]

print("Loading model...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)
print("Model loaded!")

def zero_shot_tag(ticket):
    result = classifier(ticket, candidate_labels=ALL_TAGS, multi_label=True)
    return list(zip(result["labels"][:3], result["scores"][:3]))

def few_shot_tag(ticket):
    result = classifier(
        ticket,
        candidate_labels=ALL_TAGS,
        hypothesis_template="This support ticket is about {}.",
        multi_label=True
    )
    return list(zip(result["labels"][:3], result["scores"][:3]))

def predict_tags(ticket):
    if not ticket.strip():
        return "Please enter a support ticket!"

    zs_tags = zero_shot_tag(ticket)
    fs_tags = few_shot_tag(ticket)

    output = "ZERO-SHOT PREDICTIONS\n"
    output += "=" * 35 + "\n"
    for i, (tag, score) in enumerate(zs_tags):
        bar = "█" * int(score * 20)
        output += f"Top {i+1}: {tag}\n"
        output += f"        {bar} {score*100:.1f}%\n"

    output += "\nFEW-SHOT PREDICTIONS\n"
    output += "=" * 35 + "\n"
    for i, (tag, score) in enumerate(fs_tags):
        bar = "█" * int(score * 20)
        output += f"Top {i+1}: {tag}\n"
        output += f"        {bar} {score*100:.1f}%\n"

    return output

demo = gr.Interface(
    fn=predict_tags,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter your support ticket here...",
        label="Support Ticket"
    ),
    outputs=gr.Textbox(
        label="Top 3 Predicted Tags",
        lines=15
    ),
    title="Auto Tagging Support Tickets",
    description="Zero-Shot vs Few-Shot LLM | DevelopersHub Internship",
    examples=[
        ["My internet connection keeps dropping every few minutes"],
        ["I was charged twice for my monthly subscription"],
        ["Application keeps crashing when I try to open it"],
        ["I cannot access my account after changing my email"],
        ["My printer is not being detected by the computer"],
        ["Microsoft Office is not activating on my new computer"],
    ]
)

demo.launch(share=True)
