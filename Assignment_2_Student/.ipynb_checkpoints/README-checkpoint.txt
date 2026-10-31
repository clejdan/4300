ASSIGNMENT 2: MODEL FORENSICS

1. Unzip the whole folder, not only the notebook.
2. Read Assignment_2_Model_Forensics.docx.
3. Open Assignment_2_Starter.ipynb inside this folder.
4. Use a Python notebook environment with the packages in requirements.txt.
5. Complete Questions 1–4 in each Part. Each question has matching code
   and/or an answer cell, with the same title and points as the Word file.

DECISION TREE HELP
The notebook includes an executable practice example of creating a tree,
fit(), predict(), predict_proba(), plot_tree(), export_text(), get_depth(),
get_n_leaves(), apply(), and training leaf counts. Part III uses supplied
fitted trees; Parts IV and V require fitting your own trees.

DATA MAP
part1: monthly electricity usage, continuous kWh target, 1600 observations.
part2: default within 12 months, label 1 = default, 3200 observations.
part3: service escalation, label 1 = escalation, 4000 observations.
part5: equipment failure, label 1 = event, 5000 observations.
All data are synthetic. Each has a fixed 60/20/20 train/validation/test split.
IDs and targets are not features. Part IV has no supplied dataset.

PROTOCOL CLARIFICATIONS
Use perturb_inputs() for uniform perturbations, precision, rounding, and
fixed 50-row sampling in Parts III and V. Rounding may add half a measurement
step to the realized change. Use seed 42 on validation and 43 on final test.
For III use native tree predictions; for V use the selected threshold on BOTH
original and perturbed rows. Use ddof=0 for training standard deviations.
The final test check in Part III is one accuracy evaluation of your locked
choice; it must not change your recommendation or model settings.

COMPATIBILITY
The fitted joblib models use scikit-learn 1.8.0. Use the pinned requirements.
If you cannot load them, rebuild_models.py recreates them using the supplied
training CSVs and original settings, not validation or test data. It also
reveals Part I identities: do not inspect it until you record your guesses.

FILES
models/: supplied fitted models for Parts I, II, III.
data/: split CSVs, feature dictionaries, coefficient/prediction exports.
forensics_tools.py: loading and fixed perturbation helpers.
rebuild_models.py: environment compatibility recovery only.
requirements.txt: tested dependencies (install Jupyter separately if needed).

SUBMIT
Executed .ipynb, and your Part IV .csv.
