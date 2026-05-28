# Study Protocol

## 1. Title
ML-driven Pre-screening for Early Detection of MASH with CKM Risk

## 2. Objective
Identify patients at high risk for MASH and compounding CKM conditions before formal diagnosis.

## 3. Population
Adults with metabolic risk factors such as obesity, type 2 diabetes, hypertension, dyslipidemia, CKD markers.

## 4. Inclusion Criteria
- Age ≥ 18
- Available liver enzymes: AST, ALT
- Available platelet count
- At least one metabolic risk factor

## 5. Exclusion Criteria
- Known advanced liver disease
- Missing critical lab values
- Acute liver injury context

## 6. Data Sources
- EHR structured data
- Lab results
- Diagnosis codes
- Clinical notes
- Synthetic/prototype dataset for hackathon proof-of-concept

## 7. Endpoints
Primary endpoint: High-risk MASH/CKM flag.
Secondary endpoints: model sensitivity, specificity, AUROC, explainability, clinical review burden reduction.

## 8. Analysis Plan
Compare baseline clinical scores with ML model.
Evaluate AUROC, recall, precision, F1.
Use SHAP for feature-level explanations.

## 9. Ethics & Privacy
No direct patient identifiers.
HIPAA/GDPR-aware handling.
Human-in-the-loop decision support only.
