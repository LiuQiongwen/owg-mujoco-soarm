# -*- coding: utf-8 -*-
import os, json, time, math, random, argparse
from dataclasses import dataclass
import numpy as np
import pybullet as p
import pybullet_data
import open3d as o3d

# -------------------- 参数结构 --------------------
@dataclass
class CamCfg:
    width: int = 448
    height: int = 448
    fov: float = 40.0
    znear: float = 0.2
    zfar: float = 2.0
    # 相机位姿（看 OWG demo 的配置，可按需改）
    center = np.array([0.05, -0.52, 1.9], dtype=float)
    target = np.array([0.05, -0.52, 0.785], dtype=float)
    up     = np.array([0.0,   1.0,   0.0  ], dtype=float)

# -------------------- 工具函数 --------------------
def pb_get_intrinsics(cam: CamCfg):
    # 以垂直 FOV 推算内参（近似，足够用于反投影）
    fy = cam.height / (2.0 * math.tan(math.radians(cam.fov) / 2.0))
    fx = fy
    cx = cam.width / 2.0
    cy = cam.height / 2.0
    return fx, fy, cx, cy

def pb_capture_depth_and_seg(cam: CamCfg, renderer=None):
    """
    选择渲染器（Tiny / HW-OpenGL），先计算 view/proj，再调用 getCameraImage。
    - 在 GUI 或已加载 EGL 插件的 DIRECT 模式下 → 用 HW-OpenGL
    - 否则 → 用 Tiny
    """
    # 先算视图/投影矩阵（避免未定义）
    view = p.computeViewMatrix(
        cam.center.tolist(),
        cam.target.tolist(),
        cam.up.tolist()
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cam.fov, aspect=float(cam.width)/cam.height,
        nearVal=cam.znear, farVal=cam.zfar
    )

    # 连接方式与 EGL 状态
    conn = p.getConnectionInfo().get("connectionMethod", None)
    has_egl = os.environ.get("PYBULLET_EGL", "0") == "1"

    # 解析 renderer 参数 → 目标 renderer_id
    if renderer in (None, "auto"):
        # DIRECT 且无 EGL → Tiny；其他情况（GUI 或 DIRECT+EGL）→ HW-OpenGL
        use_tiny = (conn == p.DIRECT and not has_egl)
        renderer_id = p.ER_TINY_RENDERER if use_tiny else p.ER_BULLET_HARDWARE_OPENGL
    elif renderer == "tiny":
        renderer_id = p.ER_TINY_RENDERER
    elif renderer == "opengl":
        # 只有在 GUI 或已加载 EGL 时才安全使用 HW-OpenGL，否则回退 Tiny
        if conn == p.DIRECT and not has_egl:
            print("[WARN] No EGL in DIRECT: forcing Tiny renderer")
            renderer_id = p.ER_TINY_RENDERER
        else:
            renderer_id = p.ER_BULLET_HARDWARE_OPENGL
    else:
        renderer_id = p.ER_TINY_RENDERER

    print("[INFO] Capturing frame with renderer =",
          "TINY" if renderer_id == p.ER_TINY_RENDERER else "HW-OpenGL")

    # 抓图
    w, h, rgb, depth, seg = p.getCameraImage(
        cam.width, cam.height, view, proj,
        renderer=renderer_id
    )
    depth = np.array(depth, dtype=np.float32)
    # 反投影成真实 Z
    z = cam.zfar * cam.znear / (cam.zfar - (cam.zfar - cam.znear) * depth + 1e-8)
    seg = np.array(seg, dtype=np.int32)
    return z, seg, view, proj




def depth_to_pointcloud_world(depth_z, view, proj, cam: CamCfg):
    """
    用相机 center/target/up 手动构造外参，避免 PyBullet view 矩阵行列主序混淆。
    坐标系：OpenGL 相机系，前方 -Z；像素v向下 -> y 取负；z_cam = -z。
    """
    import numpy as np, math

    H, W = depth_z.shape
    fx, fy, cx, cy = pb_get_intrinsics(cam)

    # 像素网格
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)

    z = depth_z.reshape(-1).astype(np.float32)          # z > 0（相机前方距离）
    x = (u.reshape(-1) - cx) * z / fx                   # 像素右为正
    y = -(v.reshape(-1) - cy) * z / fy                  # 像素下为负 -> 上为正
    z_cam = -z                                          # OpenGL: 前方是 -Z

    pts_cam = np.stack([x, y, z_cam], axis=1)           # 相机坐标

    # ===== 手动构造相机外参（OpenGL LookAt）=====
    eye    = cam.center.astype(np.float32)
    target = cam.target.astype(np.float32)
    up0    = cam.up.astype(np.float32)

    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)                  # 前向
    s = np.cross(f, up0); s /= (np.linalg.norm(s) + 1e-8) # 右向
    u2 = np.cross(s, f)                                 # 真正的上向

    # OpenGL 视图矩阵（世界->相机）
    V = np.array([
        [ s[0],  s[1],  s[2], -np.dot(s, eye)],
        [ u2[0], u2[1], u2[2], -np.dot(u2,eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [ 0.0,   0.0,   0.0,   1.0]
    ], dtype=np.float32)

    T_wc = np.linalg.inv(V)                              # 相机->世界

    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0],1), np.float32)], axis=1)
    pts_w_h   = (T_wc @ pts_cam_h.T).T
    pts_w     = pts_w_h[:, :3]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_w))
    return pcd



def remove_table_plane(pcd: o3d.geometry.PointCloud, voxel=0.004, dist=0.005, ransac_n=3, iters=400):
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(50)
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=ransac_n, num_iterations=200)
    # 取“非平面”作为物体点云
    obj_pcd = pcd.select_by_index(inliers, invert=True)
    return obj_pcd

def uniform_sample_indices(N, k):
    if k >= N: return np.arange(N)
    return np.random.choice(N, size=k, replace=False)

def rot_from_normal(n: np.ndarray, yaw_rad: float):
    """
    让抓取末端 -Z 轴 对齐到 法线方向 n，绕 n 再旋转 yaw（手指开合方向旋转）
    返回 R 的 ZYX 欧拉 (roll,pitch,yaw)
    """
    n = n / (np.linalg.norm(n) + 1e-8)
    # 找到把 -Z 对齐到 n 的旋转
    z_axis = np.array([0,0,-1.0], dtype=float)
    v = np.cross(z_axis, n)
    c = np.dot(z_axis, n)
    if np.linalg.norm(v) < 1e-6:
        R_align = np.eye(3)
        if c < 0:  # 180 度
            R_align = np.diag([1,-1,-1])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)
        R_align = np.eye(3) + vx + vx@vx * (1.0/(1.0+c+1e-8))
    # 绕 n 的旋转
    k = n
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], dtype=float)
    R_yaw = np.eye(3) + math.sin(yaw_rad)*K + (1-math.cos(yaw_rad))*(K@K)
    R = R_yaw @ R_align

    # 取 ZYX 欧拉
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-6:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    else:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    return roll, pitch, yaw

def quick_clearance_score(pt, n, pcd, finger_half=0.01, depth=0.03):
    """
    估个很快的“是否夹得下”的清隙分。
    在法线两侧 finger_half 的方向，局部邻域点与手指盒子的最小距离近似评分。
    """
    # 用法线的正交基构造手指横向
    n = n / (np.linalg.norm(n)+1e-8)
    tmp = np.array([1,0,0], dtype=float)
    if abs(np.dot(tmp,n)) > 0.9: tmp = np.array([0,1,0], dtype=float)
    x_dir = np.cross(n, tmp); x_dir /= (np.linalg.norm(x_dir)+1e-8)
    left  = pt + x_dir * finger_half
    right = pt - x_dir * finger_half

    # 采样邻域点——用 KDTree 最近邻距离近似
    pnts = np.asarray(pcd.points)
    if len(pnts) == 0: return 0.0
    # 为了快，随机取 2000 点做近似
    idx = uniform_sample_indices(len(pnts), 2000)
    sub = pnts[idx]
    dl = np.min(np.linalg.norm(sub - left, axis=1))
    dr = np.min(np.linalg.norm(sub - right, axis=1))
    # 手指深度方向不细抠，给个软阈
    return max(0.0, min(1.0, (dl+dr) / (2*max(1e-3, finger_half))))

def gravity_lever_score(pt, com, n):
    """越靠近质心越好，且抓取法线与“抗重力方向”一致性越好"""
    lever = np.linalg.norm(pt - com)
    lever_s = math.exp(-lever / 0.1)  # 10cm 衰减
    align = max(0.0, np.dot(n/ (np.linalg.norm(n)+1e-8), np.array([0,0,1.0]))) # 向上更好
    return 0.6*lever_s + 0.4*align

# -------------------- 主流程：生成抓取 --------------------
def generate_grasps(args):
    random.seed(args.seed); np.random.seed(args.seed)
    TABLE_BASE_Z = -0.63   # pybullet 的桌子基座 z
    TABLE_TOP_Z  = -0.004  # 运行时检测/经验得到的桌面 z（你的日志里就是这个）

    # 1) 连接仿真并摆台面+物体（和验证器保持一致）
    def connect_headless(renderer="auto", vis=0):
        if vis:
            return p.connect(p.GUI)

        cid = p.connect(p.DIRECT)
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        except Exception:
            pass

        # 尝试加载 EGL 插件（列出两种加载方式，任选成功一个）
        import pkgutil
        egl_loader = pkgutil.get_loader('eglRenderer')
        plugin_id = -1
        try:
            if egl_loader:
                plugin_id = p.loadPlugin(egl_loader.get_filename(), "_eglRendererPlugin")
            else:
                plugin_id = p.loadPlugin("eglRendererPlugin")
        except Exception:
            plugin_id = -1

        # 记录状态
        if plugin_id >= 0:
            print("[INFO] EGL renderer plugin loaded:", plugin_id)
            os.environ["PYBULLET_EGL"] = "1"
        else:
            print("[WARN] EGL plugin not available, will fall back to Tiny renderer")

        return cid


    cid = connect_headless(renderer=args.renderer if hasattr(args, "renderer") else "tiny",
                       vis=getattr(args, "vis", 0))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5,0,-0.63])
    # 物体
    obj_id = p.loadURDF(args.obj, basePosition=[0.38, 0.00, -0.63 + 0.002 + 0.5*args.cube_scale],
                        globalScaling=args.cube_scale)
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    
    # 让物体和接触稳定一下，再拍
    for _ in range(240):
        p.stepSimulation()

    # 2) 相机抓深度 → 点云（去桌面）
    cam = CamCfg(width=args.img[0], height=args.img[1])
    # 兜底：太大可能卡 -> 强制不超过 320
    cam.width = min(cam.width, 320)
    cam.height = min(cam.height, 320)

    if args.renderer == "tiny":
        renderer = p.ER_TINY_RENDERER
    elif args.renderer in ("opengl", "egl"):
        renderer = p.ER_BULLET_HARDWARE_OPENGL
    else:
        renderer = None  # auto: 在 DIRECT 下用 TINY，在 GUI 下用 OpenGL
    
    depth, seg, view, proj = pb_capture_depth_and_seg(cam, renderer)
    pcd = depth_to_pointcloud_world(depth, view, proj, cam)
    obj_pcd = remove_table_plane(pcd, voxel=args.voxel)

    # 3) 点云准备：估法线（如果 segment 后法线丢失）
    if not obj_pcd.has_normals():
        obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        obj_pcd.orient_normals_consistent_tangent_plane(50)

    pts = np.asarray(obj_pcd.points)
    nrm = np.asarray(obj_pcd.normals)
    if len(pts) == 0:
        print("[WARN] object point cloud empty, fall back to top-down")
        return fallback_topdown(args)

    # 估个物体质心（点云均值近似）
    com = np.mean(pts, axis=0)

    # 4) 采样候选
    cand = []
    base_k = min(len(pts), args.n_cand)
    pick_idx = uniform_sample_indices(len(pts), base_k)
    yaws = np.linspace(-math.pi, math.pi, args.yaw_samples, endpoint=False)

    for idx in pick_idx:
        p0 = pts[idx].copy()
        n0 = nrm[idx].copy()
        # 先抬离表面一点（避免一上来就穿模）
        n0u = n0 / (np.linalg.norm(n0)+1e-8)
        p0_lift = p0 + max(0.02, args.approach_clear) * n0u

        # 再强制一个最小 z（方块顶面上方 2cm）
        cube_top = TABLE_TOP_Z + 0.5*args.cube_scale + 0.002   # 0.002 是方块与桌面间薄片
        min_safe = cube_top + 0.020
        if p0_lift[2] < min_safe:
            p0_lift[2] = min_safe

        TABLE_Z = -0.004          # 来自 validate 的检测值
        cube_top = TABLE_Z + 0.002 + 0.5*args.cube_scale
        min_safe = cube_top + 0.020  # 方块上方 2cm

        p0_lift = p0 + max(0.02, args.approach_clear) * (n0 / (np.linalg.norm(n0)+1e-8))
        p0_lift[2] = max(p0_lift[2], min_safe)

        # 清隙快速分
        clear_s = quick_clearance_score(p0, n0, obj_pcd, finger_half=args.finger_half, depth=args.finger_depth)
        if clear_s < 0.2:
            continue
        for yaw in yaws:
            r,pit,yw = rot_from_normal(n0, yaw)
            score_g = gravity_lever_score(p0, com, n0)
            score = 0.5*clear_s + 0.5*score_g
            cand.append({
                "position": [float(p0_lift[0]), float(p0_lift[1]), float(p0_lift[2])],
                "rpy": [float(r), float(pit), float(yw)],
                "width": float(2*args.finger_half),
                "score": float(score)
            })

    if len(cand) == 0:
        print("[WARN] no candidates after clearance check, fallback")
        return fallback_topdown(args)

    # 5) 选 Top-M 进入 PyBullet 细筛（碰撞+IK 可达）
    cand = sorted(cand, key=lambda x: x["score"], reverse=True)[:max(args.topk_bullet*3, args.topk)]
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0,0,0], useFixedBase=True)

    def ik_reachable(pos, rpy, ee_index=11, iters=400):
        quat = p.getQuaternionFromEuler(rpy)
        jts = p.calculateInverseKinematics(panda_id, ee_index, pos, quat,
                                           solver=p.IK_DLS, maxNumIterations=iters, residualThreshold=1e-4)
        if jts is None: return False
        # 快速自碰/环境碰撞检测可省略一版；这里只做“能算出 IK 就行”的粗可达
        return True

    cand2 = []
    for g in cand:
        if ik_reachable(g["position"], g["rpy"], ee_index=args.ee_index, iters=args.ik_iters):
            cand2.append(g)
        if len(cand2) >= args.topk_bullet:
            break

    if len(cand2) == 0:
        print("[WARN] IK all failed, fallback topdown")
        return fallback_topdown(args)

    # 6) 输出 JSON（给验证器用）
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cand2[:args.topk], f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote {len(cand2[:args.topk])} grasps → {args.out}")

    p.disconnect()

def fallback_topdown(args):
    # 和你现在验证器的兜底一致：围绕物体上方一圈 yaw
    z_above = -0.63 + 0.002 + 0.5*args.cube_scale + 0.12
    yaw_list = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    grasps = [{
        "position": [0.38, 0.00, float(z_above)],
        "rpy": [float(np.pi), 0.0, float(yaw)],
        "width": float(2*args.finger_half),
        "score": 0.01
    } for yaw in yaw_list]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(grasps[:args.topk], f, indent=2, ensure_ascii=False)
    print(f"[INFO] fallback topdown → {args.out}")

# -------------------- CLI --------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=str, default="cube.urdf")
    ap.add_argument("--cube-scale", type=float, default=0.08)
    ap.add_argument("--out", type=str, default="grasp_6dof/dataset/sample_grasps.json")

    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--n-cand", type=int, default=1200)
    ap.add_argument("--yaw-samples", type=int, default=16)
    ap.add_argument("--finger-half", type=float, default=0.011, help="单指半宽，决定清隙检查")
    ap.add_argument("--finger-depth", type=float, default=0.03)
    ap.add_argument("--approach-clear", type=float, default=0.010)

    ap.add_argument("--topk", type=int, default=50, help="最终输出给验证器的候选数量")
    ap.add_argument("--topk-bullet", type=int, default=120, help="进入 IK 细筛的数量（会先取更大的 Top 再过滤）")
    ap.add_argument("--ee-index", type=int, default=11)
    ap.add_argument("--ik-iters", type=int, default=400)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--renderer", choices=["auto","tiny","opengl","egl"], default="auto")
    ap.add_argument("--img", type=int, nargs=2, default=[448,448], help="camera width height")
    ap.add_argument("--vis", type=int, default=0)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    generate_grasps(args)

