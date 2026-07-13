#/bin/bash

# python generate_opendihu_mu_case_package.py
cp opendihu_distribution_generator/opendihu_mu_case_package/healthy/opendihu_variables_patch_healthy.py variables/opendihu_variables_patch_healthy.py
cp opendihu_distribution_generator/opendihu_mu_case_package/death_25/opendihu_variables_patch_death_25.py variables/opendihu_variables_patch_death_25.py
cp opendihu_distribution_generator/opendihu_mu_case_package/death_50/opendihu_variables_patch_death_50.py variables/opendihu_variables_patch_death_50.py
cp opendihu_distribution_generator/opendihu_mu_case_package/death_75/opendihu_variables_patch_death_75.py variables/opendihu_variables_patch_death_75.py
# python make_protocol_B_compensated_drive.py --input-dir opendihu_mu_case_package
cp opendihu_distribution_generator/protocol_B_compensated_drive/healthy/opendihu_variables_patch_protocol_B_healthy.py variables/opendihu_variables_patch_protocol_B_healthy.py
cp opendihu_distribution_generator/protocol_B_compensated_drive/death_25/opendihu_variables_patch_protocol_B_death_25.py variables/opendihu_variables_patch_protocol_B_death_25.py
cp opendihu_distribution_generator/protocol_B_compensated_drive/death_50/opendihu_variables_patch_protocol_B_death_50.py variables/opendihu_variables_patch_protocol_B_death_50.py
cp opendihu_distribution_generator/protocol_B_compensated_drive/death_75/opendihu_variables_patch_protocol_B_death_75.py variables/opendihu_variables_patch_protocol_B_death_75.py
 
cd build_release 

./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_healthy.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_healthy.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_25.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_50.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_75.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_25.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_50.py
./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_75.py

cd ..

python tools/compare_electrodes_four_cases.py
python tools/compare_electrodes_four_cases_protocol_B.py
python tools/publication_style_emg_figures.py --csv emg_overlay_by_electrode/metrics_by_electrode.csv --out-dir metric_plots
python tools/publication_style_emg_figures.py --csv emg_overlay_by_electrode_protocol_B/metrics_by_electrode.csv --out-dir metric_plots_protocol_B

python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_50/electrodes.csv --stimulation-log build_release/out/death_50/stimulation.log --out-dir build_release/out/death_50/viz_death_50
python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_25/electrodes.csv --stimulation-log build_release/out/death_25/stimulation.log --out-dir build_release/out/death_25/viz_death_25
python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_75/electrodes.csv --stimulation-log build_release/out/death_75/stimulation.log --out-dir build_release/out/death_75/viz_death_75
python tools/visualize_opendihu_emg_enhanced.py build_release/out/healthy/electrodes.csv --stimulation-log build_release/out/healthy/stimulation.log --out-dir build_release/out/healthy/viz_healthy 

python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_50_protocol_B/electrodes.csv --stimulation-log build_release/out/death_50_protocol_B/stimulation.log --out-dir build_release/out/death_50_protocol_B/viz_death_50_protocol_B
python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_25_protocol_B/electrodes.csv --stimulation-log build_release/out/death_25_protocol_B/stimulation.log --out-dir build_release/out/death_25_protocol_B/viz_death_25_protocol_B
python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_75_protocol_B/electrodes.csv --stimulation-log build_release/out/death_75_protocol_B/stimulation.log --out-dir build_release/out/death_75_protocol_B/viz_death_75_protocol_B
python tools/visualize_opendihu_emg_enhanced.py build_release/out/healthy_protocol_B/electrodes.csv --stimulation-log build_release/out/healthy_protocol_B/stimulation.log --out-dir build_release/out/healthy_protocol_B/viz_healthy_protocol_B 

python tools/analyze_protocol_ab_same_out.py --out-root build_release/out/ --out-dir protocol_AB_analysis