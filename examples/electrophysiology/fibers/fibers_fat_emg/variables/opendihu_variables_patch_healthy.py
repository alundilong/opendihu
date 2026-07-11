# Patch for OpenDiHu variables.py. Remove/comment the original motor_units loop.
import json
import os

scenario_name = "healthy"

Conductivity = 3.828      # [mS/cm] sigma, conductivity

generated_input_directory = os.environ.get(
    "OPENDIHU_GENERATED_MU_DIR",
    r"/home/ymaom/workspace/opendihu/examples/electrophysiology/input/opendihu_distribution_generator_v2_case_package/opendihu_mu_case_package_fixed/healthy"
)

firing_times_file = os.path.join(generated_input_directory, "MU_firing_times_always.txt")
fiber_distribution_file = os.path.join(generated_input_directory, "MU_fibre_distribution_37x37_20.txt")
motor_units_file = os.path.join(generated_input_directory, "motor_units.json")

with open(motor_units_file, "r") as f:
    motor_units_payload = json.load(f)

motor_units = motor_units_payload["motor_units"]
n_motor_units = len(motor_units)

# The OpenDiHu example usually passes mu_no as a 0-based index to callbacks.
# Keep default 0. If your callback receives 1..20, set:
#   export OPENDIHU_CALLBACK_MU_INDEX_BASE=1
callback_mu_index_base = int(os.environ.get("OPENDIHU_CALLBACK_MU_INDEX_BASE", "0"))

# timing parameters
# -----------------
end_time = 100.0                    # [ms] end time of the simulation
stimulation_frequency = 100*1e-3    # [ms^-1] sampling frequency of stimuli in firing_times_file, in stimulations per ms, number before 1e-3 factor is in Hertz.
stimulation_frequency_jitter = 0    # [-] jitter in percent of the frequency, added and substracted to the stimulation_frequency after each stimulation
dt_0D = 2.5e-3                      # [ms] timestep width of ODEs (2e-3)
dt_1D = 6.25e-4                     # [ms] timestep width of diffusion (4e-3)
dt_splitting = 2.5e-3               # [ms] overall timestep width of strang splitting (4e-3)
dt_3D = 5e-1                        # [ms] time step width of coupling, when 3D should be performed, also sampling time of monopolar EMG
output_timestep_fibers = 10          # [ms] timestep for fiber output, 0.5
output_timestep_3D_emg = 10         # [ms] timestep for output big files of 3D EMG, 100
output_timestep_surface = 10        # [ms] timestep for output surface EMG, 0.5
output_timestep_electrodes = 2e8    # [ms] timestep for python callback, which is electrode measurement output, has to be >= dt_3D

# input files
input_directory = os.path.join(os.environ["OPENDIHU_HOME"], "examples/electrophysiology/input")
fiber_file              = input_directory+"/left_biceps_brachii_37x37fibers.bin"
fat_mesh_file           = fiber_file + "_fat.bin"

# stride for sampling the 3D elements from the fiber data
# a higher number leads to less 3D elements
sampling_stride_x = 2
sampling_stride_y = 2
sampling_stride_z = 40      # good values: divisors of 1480: 1480 = 1*1480 = 2*740 = 4*370 = 5*296 = 8*185 = 10*148 = 20*74 = 37*40

# HD-EMG electrode parameters
fiber_file_for_hdemg_surface = fat_mesh_file    # use the fat mesh for placing electrodes, this option is the file of the 2D mesh on which electrode positions are set
hdemg_electrode_faces = ["1+"]                  # which faces of this 2D mesh should be considered for placing the HD-EMG electrodes (list of faces, a face is one of "0-" (left), "0+" (right), "1-" (front), "1+" (back))

# xy-direction = across muscle, z-direction = along muscle
hdemg_electrode_offset_xy = 2.0           # [cm] offset from boundary of 2D mesh where the electrode array begins
hdemg_inter_electrode_distance_z = 0.4    # [cm] distance between electrodes ("IED") in z direction (direction along muscle)
hdemg_inter_electrode_distance_xy = 0.4   # [cm] distance between electrodes ("IED") in transverse direction
hdemg_n_electrodes_z = 32           # number of electrodes in z direction (direction along muscle)
hdemg_n_electrodes_xy = 12          # number of electrode across muscle

# other options
paraview_output = True
adios_output = False
exfile_output = False
python_output = False
disable_firing_output = True
enable_surface_emg = True          # Enables the surface emg output writer
fast_monodomain_solver_optimizations = True # enable the optimizations in the fast multidomain solver
optimization_type = "vc"            # the optimization_type used in the cellml adapter, "vc" uses explicit vectorization

def _mu(mu_no):
  return motor_units[(int(mu_no) - callback_mu_index_base) % len(motor_units)]

def get_am(fiber_no, mu_no):
  r = _mu(mu_no)["radius"]*1e-4
  return 2./r

def get_cm(fiber_no, mu_no):
  return _mu(mu_no)["cm"]

def get_conductivity(fiber_no, mu_no):
  return Conductivity

def get_specific_states_call_frequency(fiber_no, mu_no):
  return _mu(mu_no)["stimulation_frequency"]*1e-3

def get_specific_states_frequency_jitter(fiber_no, mu_no):
  return _mu(mu_no)["jitter"]

def get_specific_states_call_enable_begin(fiber_no, mu_no):
  return _mu(mu_no)["activation_start_time"]*1e3

