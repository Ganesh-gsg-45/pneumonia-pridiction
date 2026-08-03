---
title: PneumoVision
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🫁 PneumoVision: Pneumonia Detection from Chest X-Ray Images

An AI-powered medical image analysis system combining a custom TensorFlow CNN for pneumonia detection with Groq Cloud AI for an intelligent medical assistant.

## 🚀 Key Features

- **TensorFlow CNN** for instant chest X-ray image analysis & risk assessment
- **Groq Cloud AI (LLaMA 3.3 70B)** for medical information chatbot
- **User Authentication** via Email/Password and Google OAuth 2.0
- **PostgreSQL / SQLite Database Support** for user management

## ⚙️ Environment Variables / Secrets Required on Hugging Face

To run this Space properly, configure the following **Secrets** under your Space **Settings -> Variables and secrets**:

| Secret Name | Description |
|---|---|
| `GROQ_API_KEY` | API Key from [Groq Console](https://console.groq.com) |
| `SECRET_KEY` | Flask session secret key |
| `GOOGLE_CLIENT_ID` | Google OAuth Client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth Client Secret |
| `DATABASE_URL` | *(Optional)* PostgreSQL Connection String (falls back to SQLite) |

## 🔑 Google OAuth Redirect URI

If using Google Sign-In, add this callback URL to your **Google Cloud Console Authorized Redirect URIs**:
`https://ganesh-gsg-45-pneumonia-detection.hf.space/login/google/callback`

## ⚠️ Medical Disclaimer

This is an educational tool only. Not a medical device. Always consult qualified healthcare professionals.
