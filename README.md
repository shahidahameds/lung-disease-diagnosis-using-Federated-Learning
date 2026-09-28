# Lung Disease FL Experiment Prototype

This is the minimal working experimental prototype for the IEEE conference paper:
"Blockchain-Audited Federated Ensemble Learning with Explainable AI for Lung Disease Diagnosis".

## 1. Installation
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

## 2. Dataset Placement
Place your lung disease images in the `data/` folder.
You can either use a predefined train/test split:
```
data/train/<class_name>/<images>
data/test/<class_name>/<images>
```
Or just provide the class folders directly, and the system will split them automatically:
```
data/<class_name>/<images>
```

## 3. Configuration
Adjust hyperparameters, clients, and privacy configurations in `config.py`.

## 4. Phase 1: Inspect Dataset
To inspect the environment and dataset, generating the `dataset_summary.csv` and class distribution plot:
```bash
python main.py --inspect
```

## 5. Smoke Test
Run a minimal test with 1 communication round and 1 epoch:
```bash
python main.py --smoke-test
```

## 6. Running Individual Experiments
```bash
python main.py --experiment centralized
python main.py --experiment basic_fl
python main.py --experiment fl_dp
python main.py --experiment hierarchical
python main.py --experiment proposed
```

## 7. Running All Experiments
```bash
python main.py --experiment all
```

## 8. Output Files
Results will be generated inside the `results/` folder, including `.csv` metrics and `.png` XAI/convergence plots.

## 9. GPU Usage
The system automatically detects and utilizes CUDA if available, falling back to CPU otherwise.

cd /d "c:\Users\SHAKIL AHAMED S\Downloads\Shahid phase 1\Shahid phase 1\lung_fl_project"
.venv\Scripts\activate.bat
streamlit run app.py
