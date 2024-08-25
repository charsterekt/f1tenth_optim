import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from f1tenth_drl.DataTools.MapData import MapData
from f1tenth_drl.Planners.TrackLine import TrackLine 

from f1tenth_drl.DataTools.plotting_utils import *


set_number = 1
id_name = "TAL"
map_name = "mco"
rep_number = 4
base = f"Data/FinalExperiment_{set_number}/"
names = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
sub_paths = [f"AgentOff_SAC_Game_{map_name}_TAL_8_{set_number}_{rep_number}/",
                f"AgentOff_SAC_TrajectoryFollower_{map_name}_TAL_8_{set_number}_{rep_number}/",
                f"AgentOff_SAC_endToEnd_{map_name}_TAL_8_{set_number}_{rep_number}/", 
                f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_{rep_number}/"]

lap_n = 0
lap_list = np.ones(4, dtype=int) * lap_n



class TestLapData:
    def __init__(self, path, lap_n=0):
        self.path = path
        
        self.vehicle_name = self.path.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        self.map_name = self.vehicle_name.split("_")[3]
        self.map_data = MapData(self.map_name)
        self.std_track = TrackLine(self.map_name, False)
        self.racing_track = TrackLine(self.map_name, True)
        
        self.states = None
        self.actions = None
        self.lap_n = lap_n

        self.load_lap_data()

    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing{self.map_name.upper()}/Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def generate_state_progress_list(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.std_track.calculate_progress_percent(pt)
            # if p < progresses[-1]: continue
            progresses.append(p)
            
        return np.array(progresses[:-1]) * 100

    def calculate_deviations(self):
        # progresses = self.generate_state_progress_list()
        pts = self.states[:, 0:2]
        deviations = []
        for pt in pts:
            _, deviation = self.racing_track.get_cross_track_heading(pt)
            deviations.append(deviation)
            
        return np.array(deviations)

    def calculate_curvature(self, n_xs=200):
        thetas = self.states[:, 4]
        xs = np.linspace(0, 100, n_xs)
        theta_p = np.interp(xs, self.generate_state_progress_list(), thetas)
        curvatures = np.diff(theta_p) / np.diff(xs)
        
        return curvatures
    
    

def curvature_profile_comparison():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    xs = np.linspace(0, 100, 200)
    curve_list = [d.calculate_curvature(200) for d in data_list]

    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names):
        ax1.plot(xs[:-1], np.abs(curve_list[n]), color=color_pallet[n], label=name)
    
    ax1.set_ylabel("Curvature (rad/m)")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)
    plt.xlim(30, 60)
    plt.gca().yaxis.set_major_locator(MultipleLocator(1.5))
    y_lim = 4
    plt.ylim(0, y_lim)
    # plt.ylim(-y_lim, y_lim)

    plt.savefig(f"{base}_Imgs/CompareCurvature_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}_Imgs/CompareCurvature_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    

def lateral_deviation():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    xs_list = [d.generate_state_progress_list() for d in data_list]
    deviation_list = [d.calculate_deviations() * 100 for d in data_list]
    
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names[1:]):
        n1 = n+1
        ax1.plot(xs_list[n1], deviation_list[n1], color=color_pallet[n], label=name)

    # for n, name in enumerate(names):
    #     ax1.plot(xs_list[n], deviation_list[n], color=color_pallet[n], label=name)

    ax1.set_ylabel("Lateral \ndeviation (cm)", fontsize=9)
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=10, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(20, 80)
    # plt.xlim(10, 50)
    plt.gca().yaxis.set_major_locator(MultipleLocator(20))

    plt.savefig(f"{base}_Imgs/LateralDeviation_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}_Imgs/LateralDeviation_{map_name.upper()}.pdf", bbox_inches='tight')
    
    
def speed_steering_plot():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    fig, axes = plt.subplots(2, 4, figsize=(9, 4), sharex=True, sharey=True)
        
    action_steering_list = [d.actions[:, 0] for d in data_list]
    state_steering_list = [d.states[:, 2] for d in data_list]
    action_speed_list = [d.actions[:, 1] for d in data_list]
    state_speed_list = [d.states[:, 3] for d in data_list]
        
    for i in range(len(data_list)):
        axes[0, i].plot(action_steering_list[i], action_speed_list[i], '.', color=color_pallet[i], alpha=0.5)
        
        axes[1, i].plot(state_steering_list[i], state_speed_list[i], '.', color=color_pallet[i], alpha=0.5)
        
        axes[0, i].set_xlim(-0.45, 0.45)
        axes[0, i].grid(True)
        axes[1, i].grid(True)
        axes[0, i].set_title(names[i])
        axes[1, i].set_xlabel("Steering Angle ")
        
    axes[0, 0].set_ylim(2, 8.5)
    axes[0, 0].set_ylabel("Action Speed (m/s)")
    axes[1, 0].set_ylabel("State Speed (m/s)")
    
    name = f"{base}_Imgs/SpeedSteering{map_name.upper()}"
    std_img_saving(name, True)    
    
def speed_steering_action_plot():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    fig, axes = plt.subplots(1, 4, figsize=(6, 1.8), sharex=True, sharey=True)
        
    action_steering_list = [d.actions[:, 0] for d in data_list]
    action_speed_list = [d.actions[:, 1] for d in data_list]
        
    for i in range(len(data_list)):
        axes[i].plot(action_steering_list[i], action_speed_list[i], '.', color=color_pallet[i], alpha=0.5)
        
        axes[i].set_xlim(-0.45, 0.45)
        axes[i].grid(True)
        axes[i].set_title(names[i], fontsize=10)
        # axes[i].set_xlabel("Steering Angle", fontsize=10)
        axes[i].xaxis.set_tick_params(labelsize=8)
        axes[i].xaxis.set_major_locator(MultipleLocator(0.3))
        
    axes[0].yaxis.set_tick_params(labelsize=8)
    axes[0].set_ylim(2, 8.5)
    axes[0].set_ylabel("Speed (m/s)", fontsize=10)
    fig.text(0.53, 0.02, "Steering angle (rad)", fontsize=10, ha='center')
    
    name = f"{base}_Imgs/SpeedSteeringActions{map_name.upper()}"
    std_img_saving(name, True)


def speed_steering_plot_hist():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    fig, axes = plt.subplots(2, 4, figsize=(9, 4), sharex=True, sharey=True)
        
    action_steering_list = [d.actions[:, 0] for d in data_list]
    state_steering_list = [d.states[:, 2] for d in data_list]
    action_speed_list = [d.actions[:, 1] for d in data_list]
    state_speed_list = [d.states[:, 3] for d in data_list]
        
    for i in range(len(data_list)):
        axes[0, i].hist(action_steering_list[i], color=color_pallet[i], alpha=0.5)
        axes[1, i].hist(state_steering_list[i], color=color_pallet[i], alpha=0.5)
        
        
        axes[0, i].set_xlim(-0.45, 0.45)
        axes[0, i].grid(True)
        axes[1, i].grid(True)
        axes[0, i].set_title(names[i])
        axes[1, i].set_xlabel("Steering Angle ")
        
    # axes[0, 0].set_ylim(2, 8.5)
    axes[0, 0].set_ylabel("Action Speed (m/s)")
    axes[1, 0].set_ylabel("State Speed (m/s)")
    
    name = f"{base}_Imgs/SpeedSteeringHist{map_name.upper()}"
    std_img_saving(name, True)


def path_overlay():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    # plt.figure(1, figsize=(6, 4))
        
    xs_list = [d.states[:, 0] for d in data_list]
    ys_list = [d.states[:, 1] for d in data_list]
      
    
    map_img = MapData(map_name)
    map_img.plot_map_img()
    xs, ys = map_img.xy2rc(map_img.t_xs, map_img.t_ys)
    # plt.plot(xs, ys, '--', color='black', alpha=0.65)
    
    for i in range(len(xs_list)-1):
    # for i in range(len(xs_list)):
        xs, ys = map_img.xy2rc(xs_list[i], ys_list[i])
        plt.plot(xs, ys, color=color_pallet[i], alpha=0.65)
    
    
    
    mco_left_limits()
    
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis('off')
        
    plt.tight_layout()
    name = f"{base}Imgs/PathOverlay{map_name.upper()}"
    std_img_saving(name, True)

    
def mco_left_limits():
    plt.xlim(20, 600)
    plt.ylim(710, 1020)



# lateral_deviation()
# curvature_profile_comparison()
# speed_steering_plot()
# speed_steering_plot_hist()
# speed_steering_action_plot()
path_overlay()