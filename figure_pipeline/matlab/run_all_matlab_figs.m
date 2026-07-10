% Regenerate EVERY publication figure (SE(3) model, v9.1) at 600 dpi PNG + vector PDF.
% Run from this matlab/ dir after the python compute stage (run_all_figures.sh) has
% written the CSVs into ../ and ../coverage/. Set COVDIR if the coverage CSVs are
% elsewhere. Each figure is independent; a failure in one does not stop the rest.
here = fileparts(mfilename('fullpath'));
if ~exist('COVDIR','var'), COVDIR = fullfile(fileparts(here),'coverage'); end

figs = { ...
  % --- conserved-distance / feature importance (full non-loop) ---
  'fig_predicted_vs_actual', 'fig_residue_importance', 'fig_shap_bar_z0', 'fig_shap_bar_z1', ...
  'fig_conservation_feature_selection', 'fig_fi_method_agreement', 'fig_ca_vs_minatom', ...
  % --- drug / selectivity ---
  'fig_per_drug_overlay', 'fig_per_drug_dispersion', 'fig_drug_offtarget', ...
  'fig_compactness_vs_isolation', 'fig_ligand_type_facets', ...
  % --- mutation / phospho ---
  'fig_significance', 'fig_map2k1_phospho', 'fig_egfr_escape', 'fig_abl1_escape', ...
  % --- extended biology ---
  'fig_novel_regions', 'fig_within_kinase_diversity', 'fig_nn_validation', ...
  % --- MD ---
  'fig_braf_md_density', 'fig_marco_fgfr2_md', ...
  % --- methods / Marco Q-series ---
  'fig_pca_scree', 'fig_marco_pca_vs_latent', 'fig_marco_clustering', ...
  'fig_marco_mahalanobis', 'fig_marco_se3_ae', 'fig_marco_spline_order' };

ok = 0; fail = {};
for i = 1:numel(figs)
    s = figs{i};
    try
        run(fullfile(here,[s '.m'])); close all; ok = ok + 1;
        fprintf('OK   %s\n', s);
    catch ME
        fail{end+1} = sprintf('%s: %s', s, ME.message); %#ok<AGROW>
        fprintf(2,'FAIL %s: %s\n', s, ME.message); close all;
    end
end
fprintf('\n=== %d/%d figures OK ===\n', ok, numel(figs));
for i = 1:numel(fail), fprintf(2,'  - %s\n', fail{i}); end
