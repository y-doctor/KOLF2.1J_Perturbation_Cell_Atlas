# KOLF_Perturbation_Atlas

Steps for default workflow:

1. psp.qc.default_qc()
2. psp.pp.clean_ntc_cells()
3. psp.pp.knockdown_qc()
4. psp.pp.normalize_log_scale()
5. psp.pp.filter_sgRNA_energy_distance()
6. psp.pp.remove_unperturbed_cells_SVM()
7. psp.pp.remove_perturbations_by_cell_threshold()
8. psp.de.differential_expression()
9. psp.utils.isolate_strong_perturbations()
10. psp.pp.normalize_log_scale(scale=False)
11. psp.utils.relative_z_normalization()
12. psp.utils.compute_hvg()
13. psp.utils.count_sgRNAs_per_perturbation()
14. psp.utils.compute_perturbation_correlation()
15. psp.pl.plot_perturbation_correlation()
16. psp.pl.plot_perturbation_correlation_kde()
