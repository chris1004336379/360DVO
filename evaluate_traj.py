import numpy as np
from dpvo.plot_utils import plot_trajectory
def ate(traj_ref, traj_est, timestamps):
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation
    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=timestamps)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]

name_list=['seq0','seq1','seq2','seq3','seq4','seq5','seq6','seq7','seq8','seq9']

for name in name_list:
    ref = 'datasets/Dataset360/'+ name +'/gt.txt'
    traj_ref = np.genfromtxt(ref, delimiter=" ")[::,[2,3,4,5,6,7,8]]
    print(traj_ref.shape)
    results = []
    for i in range(5):
        print(name + '   trial' + str(i))
        est = 'saved_trajectories/'+ name +'_' + str(i) +'.txt'
        traj_est = np.loadtxt(est, delimiter=" ")[::,[1,2,3,4,5,6,7]]
        print(traj_est.shape)
        timestamps = np.genfromtxt(ref, delimiter=" ")[::,[0]]

        result = ate(traj_ref, traj_est, timestamps)

        # plot_trajectory((traj_est, timestamps), (traj_ref, timestamps), '360',
        #                                 name+'_'+str(i)+'.png', align=True, correct_scale=True)
        
        print(result)
        results.append(result)
    avg = np.mean(results)
    print(name + '   average error is: ' + str(avg))