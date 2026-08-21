# Vigilant AI Knowledge Engine

A self-hosted AI platform for document retrieval, semantic search, structured analysis, and workflow automation.

Vigilant AI demonstrates the integration of local language models, vector databases, APIs, document-processing pipelines, and automated workflows into a unified knowledge-retrieval environment.

## Overview

Vigilant AI is designed as more than a conversational chatbot. It combines local AI inference with controlled knowledge retrieval, source validation, APIs, and workflow automation. Instead of relying only on a model’s internal knowledge, the platform retrieves current information from defined data sources and returns structured, traceable outputs. 

This architecture allows knowledge to be updated independently while preserving greater control over data, logic, and automation.

## Architecture

User / Workflow  
↓  
n8n  
↓  
FastAPI  
↓  
Retrieval Engine  
↓  
Ollama + Qdrant  
↓  
Knowledge Sources

## Technology Stack

- Python
- FastAPI
- Linux
- Docker
- Qdrant
- Ollama
- n8n
- REST APIs
- Vector Embeddings
- Semantic Search
- Retrieval-Augmented Generation
- Structured JSON Processing
- Workflow Automation

## Core Capabilities

- Natural-language document search
- Semantic vector retrieval
- Multi-source knowledge retrieval
- Local AI inference
- Structured response generation
- Source-backed analysis
- Document ingestion and processing
- Metadata enrichment
- Retrieval validation
- REST API integration
- Automated workflow orchestration
- Self-hosted AI infrastructure

## Engineering Work

Development of Vigilant AI includes:

- Deploying local AI services on Linux
- Integrating language and embedding models
- Designing vector-search workflows
- Building Python retrieval services
- Developing FastAPI endpoints
- Creating ingestion and metadata-processing pipelines
- Implementing multi-collection retrieval
- Automating workflows with n8n
- Testing retrieval quality and response validation
- Troubleshooting API, service, model, and database integration

## Project Structure

- `app/` - API services
- `config/` - Application schemas and configuration
- `scripts/` - Retrieval, ingestion, metadata, vector database, and validation utilities
- `requirements.txt` - Python dependencies

## Design Principles

- Local-first processing
- Modular services
- Traceable information retrieval
- Source-grounded responses
- Repeatable automation
- Structured outputs
- Extensible knowledge collections

## Status

Active development and technical portfolio project.

This repository demonstrates the architecture and engineering methods used to build a self-hosted AI knowledge retrieval and automation platform.
