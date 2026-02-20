
---
title: Industrial AI Safety Auditor & Governance Framework
description: Search for any action in a video without pre-training labels
emoji: 📽
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: true
---
# 🛡️ Industrial AI Safety Auditor & Governance Framework
### *Enterprise-Grade Video Analytics with Automated Safety Case Generation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AI-Powered](https://img.shields.io/badge/AI-Vision--Language--Model-orange.svg)]()

This repository contains a **Solution Architecture** for deploying Vision-Language Models (VLMs) in high-stakes, safety-critical environments. Designed with industrial safety assurance principles in mind, the system goes beyond simple detection to provide **Temporal Action Auditability**.

The framework autonomously identifies specific safety/security incidents and generates a **Safety Case Evidence (PDF)**, ensuring every AI inference is backed by a quantifiable environmental and robustness audit.

---

## 🏗️ Solution Architecture

The system follows a **Decoupled Governance Pattern**, ensuring that raw AI outputs are validated by a deterministic "Safety Gate" before being committed to an official record.

graph TD
    A[Video Ingestion] --> B{Governance Gate}
    B -- Fail: Low Light/Contrast --> C[Abort & Log Error]
    B -- Pass: Verified Conditions --> D[Decord Frame Extraction]
    D --> E[Qwen2-VL Analysis]
    E --> F[Forensic Reasoning Engine]
    F --> G[FPDF Reporting Layer]
    G --> H[Official Safety Case PDF]

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#00ff00,stroke:#333,stroke-width:2px
    style C fill:#ff0000,stroke:#333,stroke-width:2px

### **Core Pipeline Components**

1. **Ingestion Layer:** Efficient video frame extraction using `Decord` to minimize memory overhead during high-resolution processing.
2. **Phase 1: Governance Gate (Environmental Audit):** Validates site conditions (luminance and contrast thresholds) to prevent model "hallucination" in sub-optimal visibility.
3. **Phase 2: Inference Engine (Qwen2-VL):** - **Semantic Reasoning:** Generates human-readable logic explaining why a specific frame was flagged for safety violations.
4. **Phase 3: Reporting Layer:** Programmatic generation of `Safety_Case_Evidence.pdf` containing a full audit trail including timestamps, AI reasoning, and environmental metrics.

---

## 🚀 Live Demonstration Scenarios

The system is pre-configured with two "Mission-Critical" scenarios:

* [cite_start]**Asset Protection (Forensic):** Detecting unauthorized removal of equipment from a vehicle interior. [cite: 6, 15]
* **Infrastructure Safety (Urban):** Monitoring high-density junctions for compliance with transit safety protocols.

---

## 📊 Governance & Data Integrity

Unlike standard AI demos, this auditor includes a **Hardware-Software Handshake**. Before the AI processes the video, the system performs a deterministic check:

| Metric | Threshold | Purpose |
| :--- | :--- | :--- |
| **Brightness** | > 40 | [cite_start]Ensures site lighting is sufficient for object extraction. [cite: 27] |
| **Contrast** | > 15 | [cite_start]Detects lens obstruction, fog, or extreme muddying of the scene. [cite: 27] |
| **Integrity Status** | **VERIFIED** | [cite_start]Only "Verified" data is permitted to generate a Safety Case. [cite: 28] |

---

## 🛠️ Technical Stack

* **AI/ML Framework:** `transformers`, `torch`, `Qwen2-VL`
* **Orchestration:** `Gradio` (Secure Web Interface)
* **Video Engineering:** `OpenCV`, `Decord`
* [cite_start]**Compliance Reporting:** `FPDF` (Programmatic PDF Generation) [cite: 1, 2, 26]

---

## 🔧 Installation & Deployment

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/jameswhook2236-oss/AI-Safety-Auditor-Pipeline.git](https://github.com/jameswhook2236-oss/AI-Safety-Auditor-Pipeline.git)
   cd AI-Safety-Auditor-Pipeline

pip install -r requirements.txt

python app.py
