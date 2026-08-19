# sld-mcp-server

MCP server for public SLD data analysis and electroweak measurements.

This server exposes agent-callable tools for working with the public SLD release: the modernized parquet translation of the 1996–1998 polarized SLD e+e- Z-pole dataset described in *An AI-ready, Polarized Electron-Positron Collision Dataset* (arXiv:2606.00224).

This server uses the SLD Resurrection analysis codebase.

## Tool categories

This server exposes tools in four categories:
- Data Inspection
- Selection & Cutflow
- Physics Computation
- Plotting

## Current scope

This server currently supports:
- inspection of SLD parquet shards and bank structure
- selection preset discovery and cutflow reporting
- hadronic event selection and A_LR extraction
- leptonic asymmetry measurements
- visible mass, event-shape, angular, and weak-mixing summary plots

This server does not yet expose OmniLearned embedding or classifier workflows.

## Installation

Install in editable mode from the repository root:

pip install -e .

## Run

Start the MCP server with:

sld_mcp_server

## Requirements

This server assumes:
- local access to the public SLD parquet dataset
- a working installation of SLD Resurrection
- Python 3.10 or newer
