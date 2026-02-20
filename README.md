
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

Industrial AI Safety Auditor & Governance Framework

Enterprise-Grade Video Analytics with Automated Safety Case Generation

📝 Executive Summary

This repository contains a Solution Architecture for deploying Vision-Language Models (VLMs) in high-stakes, safety-critical environments. Designed with BAE Systems’ Safety Assurance principles in mind, the system goes beyond simple object detection to provide Temporal Action Auditability.

The framework autonomously identifies specific safety/security incidents and generates a Safety Case Evidence report (PDF), ensuring every AI inference is backed by a quantifiable environmental and robustness audit.

🏗 Solution Architecture

The system follows a Decoupled Governance Pattern, ensuring that raw AI outputs are validated by a "Safety Gate" before being committed to an official record.

Core Components

1. Ingestion Layer: Efficient video frame extraction using Decord to minimize memory overhead during high-resolution processing.

2. Phase 1: Governance Gate (Environmental Audit): Validates site conditions (luminance and contrast) to prevent model "hallucination" in sub-optimal visibility.

3. Phase 2: Inference Engine (X-CLIP + Qwen2.5-VL): * Temporal Localization: Identifies precise "start/stop" timestamps for specific actions.

4. Semantic Reasoning: Generates human-readable logic for why an action was flagged.

5. Phase 3: Reporting Layer: Programmatic generation of Safety_Case_Evidence.pdf containing a full audit trail of metrics and visual evidence.

🚀 Live Demonstration

The system is pre-configured with two "Mission-Critical" scenarios:

* Asset Protection (Forensic): Detecting unauthorized removal of equipment from a vehicle interior.

* Infrastructure Safety (Urban): Monitoring high-density junctions for compliance with transit safety protocols.

🛠 Technical Stack

AI/ML: transformers, torch, X-CLIP, Qwen2.5-VL

Orchestration: Gradio (Web Interface & API)

Video Processing: OpenCV, MoviePy, Decord

Compliance Reporting: FPDF (Automated PDF Generation)
