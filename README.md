# Computational Thermodynamics — Binary Eutectic System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmed-elnaml/computational-thermo/blob/main/eutectic_colab.ipynb)

> **For supervisors / reviewers:** Click the badge above → the notebook opens in Google Colab in your browser. No installation required. Press **Runtime → Run all**.

---

## What this project does

A CALPHAD-based interactive tool for computing and visualising:

| Feature | Details |
|---------|---------|
| Free-energy G–x curves | With **correct common-tangent construction** and contact-point markers |
| Phase diagram | Binary T–x diagram with live **temperature indicator** overlaid |
| Thermo properties | Enthalpy, Entropy, and Thermodynamic Activity |

---

## Option A — Google Colab (zero installation)

1. Click the **Open in Colab** badge at the top of this README.  
2. In the notebook, press **Runtime → Run all**.  
3. Wait ~30 s for the phase diagram to compute.  
4. Use the sliders to explore.

To use your own `.tdb` database file: select *"Upload custom .tdb"* from the dropdown and upload your file.

---

## Option B — Run locally (Python required)

```bash
pip install -r requirements.txt
python eutectic_thermo.py
```

The script auto-detects any `.tdb` files in the folder and shows a numbered menu.

---

## Repository contents

| File | Purpose |
|------|---------|
| `eutectic_colab.ipynb` | **Colab notebook** — run in browser, no install |
| `eutectic_thermo.py` | Standalone desktop script |
| `alzn_mey.tdb` | Al-Zn thermodynamic database (Mey 1993) |
| `requirements.txt` | Python dependencies for local use |

---

## Updating the Colab badge

After creating your GitHub repository, replace `YOUR_GITHUB_USERNAME` and `YOUR_REPO_NAME` in this README with your actual values. The badge will then link directly to the notebook.
