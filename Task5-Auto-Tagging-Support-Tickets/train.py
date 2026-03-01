import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score
import torch

# Dataset
tickets = [
    "My internet connection keeps dropping every few minutes",
    "WiFi is very slow and pages are not loading properly",
    "Cannot connect to the network since this morning",
    "VPN is not working and I cannot access company resources",
    "Network speed is extremely slow during peak hours",
    "I was charged twice for my monthly subscription",
    "My invoice shows wrong amount please correct it",
    "I want to cancel my subscription and get a refund",
    "Payment failed but money was deducted from my account",
    "I need a receipt for my last payment for tax purposes",
    "Application keeps crashing when I try to open it",
    "Software update failed and now program is not working",
    "I cannot login to my account password reset is not working",
    "The mobile app is freezing on the loading screen",
    "Error message appears every time I try to save my work",
    "I cannot access my account after changing my email",
    "My account was suspended without any warning or reason",
    "I need to update my personal information in the system",
    "Two factor authentication is not sending the code to my phone",
    "I forgot my username and cannot recover my account",
    "My printer is not being detected by the computer",
    "Laptop screen is flickering and showing strange colors",
    "Keyboard keys are not responding when pressed",
    "Computer is making loud noise and running very hot",
    "External hard drive is not showing up on my computer",
    "Microsoft Office is not activating on my new computer",
    "Antivirus software is blocking my legitimate applications",
    "Browser extensions are causing conflicts with the website",
    "Operating system update broke my audio drivers",
    "Database software is throwing SQL connection errors",
]

labels = [
    "Network Issue", "Network Issue", "Network Issue",
    "Network Issue", "Network Issue",
    "Billing Issue", "Billing Issue", "Billing Issue",
    "Billing Issue", "Billing Issue",
    "Technical Support", "Technical Support", "Technical Support",
    "Technical Support", "Technical Support",
    "Account Issue", "Account Issue", "Account Issue",
    "Account Issue", "Account Issue",
    "Hardware Issue", "Hardware Issue", "Hardware Issue",
    "Hardware Issue", "Hardware Issue",
    "Software Issue", "Software Issue", "Software Issue",
    "Software Issue", "Software Issue",
]

df = pd.DataFrame({"ticket": tickets, "label": labels})
print(f"Dataset shape: {df.shape}")
print(df["label"].value_counts())

ALL_TAGS = [
    "Network Issue", "Billing Issue", "Technical Support",
    "Account Issue", "Hardware Issue", "Software Issue",
]

print("Loading classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

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

print("Evaluating...")
zero_shot_preds = [zero_shot_tag(t)[0][0] for t in df["ticket"]]
few_shot_preds  = [few_shot_tag(t)[0][0]  for t in df["ticket"]]

zs_acc = accuracy_score(df["label"], zero_shot_preds)
fs_acc = accuracy_score(df["label"], few_shot_preds)

print(f"Zero-Shot Accuracy : {zs_acc * 100:.2f}%")
print(f"Few-Shot Accuracy  : {fs_acc * 100:.2f}%")
