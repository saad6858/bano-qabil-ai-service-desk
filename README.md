# Bano Qabil AI Service Desk

A multi-channel, multi-agent AI service desk designed for Bano Qabil.

## Project Architecture

The system is designed around separate capabilities for different types of information:

- General Bano Qabil information → RAG / website knowledge
- Course curriculum information → curriculum RAG
- Live class schedules → calendar service
- Private student information → separate student database
- Unresolved or special cases → human escalation by email

The system will eventually support Website, WhatsApp, and Email as user-facing channels.

## Development Environment

This project is developed in GitHub Codespaces.

The Python implementation uses Python 3.13.x and uv for isolated dependency management and reproducible dependency locking.

The exact Python version is pinned in `.python-version`.

The resolved dependency versions will be stored in `uv.lock`.

## Important Data Separation

Private student records must never be inserted into the public knowledge/vector databases.

Schedule information must not be stored as RAG knowledge.

The project uses separate storage mechanisms for:

1. Website/general knowledge
2. Curriculum knowledge
3. Private student records
4. Live schedule information

## Current Development Stage

Phase 1 — Foundation.

No production credentials or real student data should be committed to this repository.
