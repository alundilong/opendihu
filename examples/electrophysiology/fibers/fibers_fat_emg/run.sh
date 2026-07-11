#/bin/bash
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_healthy.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_healthy.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_25.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_50.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_75.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_25.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_50.py
#mpirun -np 1 ./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_75.py

#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_healthy.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_healthy.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_25.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_50.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_death_75.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_25.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_50.py
#./fibers_fat_emg ../settings_fibers_fat_emg2.py opendihu_variables_patch_protocol_B_death_75.py

# python tools/compare_electrodes_four_cases.py
# python tools/compare_electrodes_four_cases_protocol_B.py
# python tools/publication_style_emg_figures.py --csv emg_overlay_by_electrode/metrics_by_electrode.csv --out-dir metric_plots
# python tools/publication_style_emg_figures.py --csv emg_overlay_by_electrode_protocol_B/metrics_by_electrode.csv --out-dir metric_plots_protocol_B

# python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_50/electrodes.csv --stimulation-log build_release/out/death_50/stimulation.log --out-dir viz_death_50
# python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_25/electrodes.csv --stimulation-log build_release/out/death_25/stimulation.log --out-dir viz_death_25
# python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_75/electrodes.csv --stimulation-log build_release/out/death_75/stimulation.log --out-dir viz_death_75
# python tools/visualize_opendihu_emg_enhanced.py build_release/out/healthy/electrodes.csv --stimulation-log build_release/out/healthy/stimulation.log --out-dir viz_healthy 

# python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_50_protocol_B/electrodes.csv --stimulation-log build_release/out/death_50_protocol_B/stimulation.log --out-dir viz_death_50_protocol_B
# python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_25_protocol_B/electrodes.csv --stimulation-log build_release/out/death_25_protocol_B/stimulation.log --out-dir viz_death_25_protocol_B
# python tools/visualize_opendihu_emg_enhanced.py build_release/out/death_75_protocol_B/electrodes.csv --stimulation-log build_release/out/death_75_protocol_B/stimulation.log --out-dir viz_death_75_protocol_B
# python tools/visualize_opendihu_emg_enhanced.py build_release/out/healthy_protocol_B/electrodes.csv --stimulation-log build_release/out/healthy_protocol_B/stimulation.log --out-dir viz_healthy_protocol_B 

# python tools/analyze_protocol_ab_same_out.py --out-root build_release/out/ --out-dir protocol_AB_analysis
# 