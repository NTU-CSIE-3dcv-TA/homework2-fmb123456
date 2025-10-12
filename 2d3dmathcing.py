from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using OpenCV
    # Hint: you may use "Descriptors Matching and ratio test" first
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_query, desc_model, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts_query = np.array([kp_query[m.queryIdx] for m in good], dtype=np.float64)
    pts_model = np.array([kp_model[m.trainIdx] for m in good], dtype=np.float64)

    return cv2.solvePnPRansac(
        objectPoints=pts_model,
        imagePoints=pts_query,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
    )

def rotation_error(R1, R2):
    #TODO: calculate rotation error
    return np.rad2deg(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))

def translation_error(t1, t2):
    #TODO: calculate translation error
    return np.linalg.norm(t1 - t2)

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    #TODO: visualize the camera pose
    xyz = np.vstack(points3D_df["XYZ"].to_list()).astype(np.float64)
    rgb = np.vstack(points3D_df["RGB"].to_list()).astype(np.float64) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    s = 0.1
    pts = [[0, 0, 0], [s, s, s], [s, -s, s], [-s, -s, s], [-s, s, s]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [[1, 0, 0]] * 8
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors=o3d.utility.Vector3dVector(colors)

    cameras = []
    traj_pts = []
    for T in Camera2World_Transform_Matrixs:
        pts = np.asarray(ls.points)
        pts_w = (np.hstack([pts, np.ones((pts.shape[0], 1))]) @ T.T)[:,:-1]
        g = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts_w),
            lines=ls.lines,
        )
        g.colors=ls.colors
        cameras.append(g)
        traj_pts.append(pts_w[0])
    traj_lines = [[i, i + 1] for i in range(len(traj_pts) - 1)]
    traj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(traj_pts),
        lines=o3d.utility.Vector2iVector(traj_lines),
    )
    traj.colors=o3d.utility.Vector3dVector([[0, 0, 1]] * len(traj_lines))

    o3d.visualization.draw_geometries(
        [pcd, traj, *cameras],
        front=[0.11830943823025981, -0.011194892767812936, -0.99291366754696131],
        lookat=[1.216128554086632, -1.0611115902268715, -4.3286562441519019],
        up=[-0.10799062623784268, -0.9941505285954122, -0.0016586555462688708],
        zoom=0.02
    )


def draw_cube(img, cube_vertices, rvec, tvec):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    faces = [
        [1, 0, 2, 3],
        [5, 7, 6, 4],
        [5, 4, 0, 1],
        [4, 6, 2, 0],
        [6, 7, 3, 2],
        [7, 5, 1, 3],
    ]
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0)
    ]

    all_world_pts = []
    all_colors = []

    for i, face in enumerate(faces):
        v0, v1, v2, v3 = cube_vertices[face]
        color = colors[i]
        for i in np.linspace(0, 1, 10):
            for j in np.linspace(0, 1, 10):
                p = (1 - i) * ((1 - j) * v0 + j * v1) + i * ((1 - j) * v3 + j * v2)
                all_world_pts.append(p)
                all_colors.append(color)

    all_world_pts = np.array(all_world_pts)
    all_colors = np.array(all_colors, dtype=np.uint8)

    pts2d, _ = cv2.projectPoints(all_world_pts, rvec, tvec, cameraMatrix, distCoeffs)

    rmat = R.from_rotvec(rvec.flatten()).as_matrix()
    all_z = (rmat @ all_world_pts.T + tvec)[-1]
    order = np.argsort(all_z)[::-1]

    pts2d = pts2d.reshape(-1, 2)[order]
    all_z = all_z[order]
    all_colors = all_colors[order]

    img_out = img.copy()
    h, w = img.shape[:2]
    for (x, y), c, z in zip(pts2d, all_colors, all_z):
        if z <= 0:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img_out, (xi, yi), 3, c.tolist(), -1)
    return img_out

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    cube_transform = np.load("cube_transform_mat.npy")
    cube_vertices = np.load("cube_vertices.npy")
    cube_vertices = (np.hstack([cube_vertices, np.ones((cube_vertices.shape[0], 1))]) @ cube_transform.T)

    IMAGE_ID_LIST = [i for i in range(164, 294)]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    img_list = []
    frameid_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        # tvec = tvec.reshape(1,3) # Reshape translation vector
        rmat = R.from_rotvec(rvec.flatten()).as_matrix()
        r_list.append(rmat)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rmat, R.from_quat(rotq_gt).as_matrix().reshape(3, 3))
        t_error = translation_error(tvec, tvec_gt.reshape(tvec.shape))
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

        img_list.append(draw_cube(rimg, cube_vertices, rvec, tvec))
        frameid = int(fname[9:-4])
        frameid_list.append(frameid)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    med_r = np.median(rotation_error_list)
    med_t = np.median(translation_error_list)
    print(f"Median Rotation Error (deg): {med_r}")
    print(f"Median Translation Error: {med_t}")

    # TODO: result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system
        c2w = np.vstack([np.hstack([r.T, -r.T @ t]), [0, 0, 0, 1]])
        Camera2World_Transform_Matrixs.append(c2w)
    Camera2World_Transform_Matrixs = np.array(Camera2World_Transform_Matrixs)[np.argsort(frameid_list)]
    visualization(Camera2World_Transform_Matrixs, points3D_df)

    cv2.namedWindow("AR video")
    cv2.moveWindow("AR video", 500, 0)
    for i in np.argsort(frameid_list):
        cv2.imshow("AR video", img_list[i])
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
