# CSC8499 Individual Project

## Exploring Optimisation Techniques for End-to-End Autonomous Racing Pipelines to Approach Perception-Planning-Control Pipeline Performance

### Details:
* Author: Dhruv R.K.
*  Student ID: 230108224
*  MSc Advanced Computer Science, Newcastle University


## Repository Explained:

This repository houses the code experiments performed from which the results in my dissertation were obtained and evaluated. The repository covers both the vision-based approaches as well as the reinforcement learning approaches described in the dissertation. This README file explains how to navigate the code and cites references to any original works.

The repository is divided into two main directories.

 `vision_experiments` contains an Ipython Notebook file from my training process on Google Colab, and was used to parse the recorded image and sensor data into manageable files, and train the baseline vision model along with all its versions. It also contains the template files for each model to communicate and work with the AutoDRIVE simulator and directories for model and real-time utils.

 `deepRL_experiments` contains a modified fork from the F1Tenth DRL framework which itself contains a fork of the F1Tenth Gym environment, and also has
  modified files to accomodate the testing methods alongside the baseline which was taken directly from the original work.

 ## Missing Files:

 `vision_experiments`: The IPython notebook used to create dataset files and models saves all data to my personal Google Drive and the files are too large to be put on GitHub directly, so all split dataset and Torch model dicts are missing. The `devkit` directory is a modified and minimal version of the dev kit provided by the AutoDRIVE ecosystem, and includes the original unmodified License file. All templates are also based on the missing example script provided in the original dev kit. The AutoDRIVE simulator itself was used as a standalone executable and is not included here. The PPC Follow-the-Gap code is a changed and different Python-only version of the algorithm proposed in a blog post cited below.

 `deepRL_experiments`: This directory is a partial fork of the original work in F1Tenth DRL, keeping all the files within the `f1tenth_drl` subdirectory from the original repo and modifying the given list of files for this project. The original project itself includes various track files from other public sources and contains a modified fork of the F1tenth Gym environment. For the sake of time, modifications were made in commented-out blocks of code that are uncommented and commented as required to change the functionality on the fly, instead of integrating these changes into the YAML templates used by the project.

 Files modified for F1Tenth DRL:

 * `Data/Experiment_1`: Contains autorecorded stats for the final set of experiments used.
 * `Experiments/Experiment.yaml` is the config that is changed for each experiment.
 * `LearningAlgorithms/td3 and LearningAlgorithms/td3_quantizable` modified to implement PTSQ/PTDQ and QAT respectively. `td3_quantizable` consolidates methods and networks found from other portions of the repo into one file for ease of use.
 * `LearningAlgorithms/create_agent` modified to accomodate the new functionalities of `td3` and `td3_quantizable`.
 * `Planners/AgentTester and Planners/AgentTrainer` modified for the same as above.
 * `run_experiments` modified for changing the running parameters and visualising the process.


## References:

* F1Tenth DRL (modified fork): https://github.com/BDEvan5/f1tenth_drl/tree/master and https://www.sciencedirect.com/science/article/pii/S266682702300049X
* F1Tenth Gym (used within F1Tenth DRL): https://github.com/f1tenth/f1tenth_gym
* AutoDRIVE Simulator Executable: https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE/tree/AutoDRIVE-Simulator
* AutoDRIVE Devkit (stripped down from): https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE/tree/AutoDRIVE-Devkit
* All PyTorch General Docs Ref: https://pytorch.org/docs/stable/index.html
* Follow-the-Gap blog post: https://lejunjiang.com/2021/01/28/f1tenth-lab-4/
