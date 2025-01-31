# KOLF_Perturbation_Atlas

Steps for default workflow:

1. psp.qc.default_qc()
2. psp.pp.clean_ntc_cells()
3. psp.pp.knockdown_qc()
4. psp.pp.normalize_log_scale()
5. psp.pp.filter_sgRNA_energy_distance()
6. psp.pp.remove_unperturbed_cells_SVM()
7. psp.pp.remove_perturbations_by_cell_threshold()
