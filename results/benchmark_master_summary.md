# MARL Benchmark Master Results Summary

> **Format Note**: This document is formatted in standard GitHub Flavored Markdown with precise tabular layouts and statistical aggregates (Mean ± Std, Interquartile Mean [IQM], and 95% Bootstrap Confidence Intervals) specifically designed for direct ingestion and reasoning by Large Language Models.

- **Run ID**: `test_run`
- **Generated At**: 2026-09-14 18:54:52 UTC
- **Evaluated Stages**: Stages 0 to 4 (Maps $11\times 11$ to $50\times 50$)

---

## Table 1: Executive Benchmark Performance Matrix

| Algorithm | Stage | Map Size | Seeds | Win Rate (%) | Win Rate (IQM %) | 95% Bootstrap CI (%) | Mean Return | Lifetime Win % | Stealth Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **THIEF** | Stage 0 | 11x11 | 2 | 0.0 ± 0.0 | 0.0 | [0.0, 0.0] | 0.00 ± 0.00 | 0.0 | 1.000 |
| **ECOOP** | Stage 0 | 11x11 | 1 | 0.0 ± 0.0 | 0.0 | [0.0, 0.0] | 0.00 ± 0.00 | 0.0 | 1.000 |

---

## Table 2: Tactical Subtask Mastery & Coordination Telemetry

| Algorithm | Stage | Scout Tag (%) | Hacker Hack (%) | Muscle Neut (%) | Extractor Loot (%) | Full Squad Extract (%) | Ghost Run (%) | Alarm Velocity | Primary Failure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **THIEF** | Stage 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 | `general_attrition` |
| **ECOOP** | Stage 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 | `general_attrition` |

---

## Table 3: Compute, Hardware & MoE Scalability Telemetry

| Algorithm | Stage | FPS Throughput | Wall Clock (s) | Peak VRAM (MB) | Active Experts | Effective Experts | Routing Entropy (nats) | Role-Expert MI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **THIEF** | Stage 0 | 98 | 1.0s | 0.7 | 1.0 | 1.00 | 0.000 | 0.000 |
| **ECOOP** | Stage 0 | 95 | 1.1s | 0.7 | 1.0 | 1.00 | 0.000 | 0.000 |
