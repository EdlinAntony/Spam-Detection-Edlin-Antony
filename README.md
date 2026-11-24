# Spam Detection & Confusion Matrix Analysis

## 📖 Project Overview
This project builds an automated email classification system to detect spam messages. It uses the **Categorical Naive Bayes** algorithm to analyze text features (like the presence of specific keywords) and evaluates the model's performance using a visual **Confusion Matrix**.

## 🎯 Objective
To automatically classify incoming emails as "Spam" or "Not Spam" to prevent misinformation and protect users, while providing clear metrics on the model's accuracy and error rates.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn (CategoricalNB), Matplotlib, Seaborn
- **Algorithm:** Categorical Naive Bayes

## 📊 Dataset
The model uses a dataset of emails with the following categorical features:
- **Offer:** Does the email contain the word "Offer"? (Yes/No)
- **Money:** Does the email contain the word "Money"? (Yes/No)
- **Meeting:** Does the email contain the word "Meeting"? (Yes/No)
- **Actual:** The true label (Spam / Not Spam)

## 🚀 How to Run
1. Ensure you have Python installed on your system.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

## Project Screenshots

<img width="1917" height="1020" alt="Screenshot 2026-11-24 134726" src="https://github.com/user-attachments/assets/3efe6f44-67a4-48db-8d41-adc3d31f6076" />
