#!/usr/bin/env python3
"""
launch_franka_scene.py
======================
使用 MuJoCo 3.x MjSpec API 将 Vention 桌子网格与 Franka Panda 机械臂组合成完整场景，
并打开 MuJoCo live viewer 窗口。

场景坐标系说明：
- 地板平面：Z = 0
- 桌底落地：Z = 0，桌面高度约 0.891 m（世界坐标）
- Franka 底座：安装在桌子 +X 短边正中央桌面上（朝向 -X，即朝桌子内侧）

STL 坐标系（原始）：X 为长度(2029mm)，Y 为高度(891mm)，Z 为深度(-1879~-652mm)
旋转修正：绕 X 轴 +90°，将 Y(高度) 映射到世界 Z(垂直)方向

STL 单位：mm，通过 scale=0.001 转换为 m

注意：MuJoCo 3.2.6 存在内部路径处理 bug：当 mesh 文件绝对路径超过约 22 字符时，
compile() 会 segfault。解决方案是设置 scene.meshdir 并在 mesh.file 中使用相对路径（仅文件名）。
"""

import os
import pathlib
import sys

import mujoco
import mujoco.viewer

# 路径定义（脚本位于 scripts/，项目根在上一级）
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

# 桌子 STL（10000 面简化版，与 panda.xml 兼容）
TABLE_STL = _MODELS_DIR / "vention_table_10k.stl"

# Franka Panda MJCF：优先使用环境变量 MUJOCO_MENAGERIE，否则使用默认路径
_MENAGERIE = pathlib.Path(
    os.environ.get(
        "MUJOCO_MENAGERIE",
        str(_PROJECT_ROOT.parent / "mujoco_menagerie")
    )
)
PANDA_XML = _MENAGERIE / "franka_emika_panda/panda.xml"


def build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    使用 MjSpec API 组合桌子网格和 Franka Panda，返回编译好的 (model, data)。

    桌子坐标变换（已预计算）：
      STL 原始范围（mm）：X[-24, 2005], Y[-116, 775], Z[-1879, -652]
      旋转 +90° 绕 X 轴：(x,y,z)_body → (x,-z,y)_world
        world_Z = body.pos.z + y_body，y_min=-0.116 → body.pos.z=0.116，桌面 Z=0+0.775+0.116≈0.891m
        world_Y = body.pos.y - z_body，z_range=[-1.879,-0.652] → y_world∈[0.652,1.879]+pos.y
          居中：body.pos.y = -(0.652+1.879)/2 = -1.2655
        world_X 不变：body.pos.x = -0.9905（X 方向居中）
      body.pos = [-0.9905, -1.2655, 0.116]
      body.quat = [0.7071, 0.7071, 0, 0]  （绕 X 轴 90°，w=cos45°, x=sin45°）
      桌面高度 = 0.116 + 0.775 = 0.891 m（世界坐标 Z）

    Franka 安装位置：
      frame.pos = [1.015, 0, 0.891]   （桌子 +X 短边正中央，桌面上）
      frame.quat = [0, 0, 0, 1]       （MuJoCo 四元数格式 w,x,y,z；绕 Z 轴 180°，朝向 -X）

    MuJoCo 3.2.6 兼容性注意：
      必须设置 scene.meshdir，并在 mesh.file 中使用相对于 meshdir 的文件名，
      否则绝对路径过长会导致 compile() segfault（内部 buffer overflow bug）。
    """
    # 加载 Franka Panda 子规格
    panda_spec = mujoco.MjSpec.from_file(str(PANDA_XML))

    # 创建场景主规格
    scene = mujoco.MjSpec()

    # 设置 meshdir，避免 MuJoCo 3.2.6 内部路径处理 bug（绝对路径过长时 segfault）
    # mesh.file 使用相对于此目录的文件名
    scene.meshdir = str(_MODELS_DIR)

    # 地板
    floor = scene.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5, 5, 0.1]
    floor.rgba = [0.6, 0.6, 0.6, 1.0]

    # 桌子网格资产（STL，mm → m）
    # 使用相对于 scene.meshdir 的文件名，避免绝对路径 bug
    table_mesh = scene.add_mesh()
    table_mesh.name = "vention_table"
    table_mesh.file = TABLE_STL.name   # 仅文件名，MuJoCo 在 meshdir 中查找
    table_mesh.scale = [0.001, 0.001, 0.001]

    # 桌子 body：
    # STL 原始坐标系 Y 为高度方向，需绕 X 轴旋转 +90° 使桌面水平
    # quat=[cos45°, sin45°, 0, 0]=[0.7071,0.7071,0,0] 对应绕 X 轴 90°
    # pos 预计算使桌底落在 Z=0，XY 居中
    table_body = scene.worldbody.add_body()
    table_body.name = "table"
    table_body.quat = [0.7071, 0.7071, 0, 0]
    table_body.pos = [-0.9905, -1.2655, 0.116]

    table_geom = table_body.add_geom()
    table_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    table_geom.meshname = "vention_table"
    table_geom.rgba = [0.75, 0.78, 0.82, 1.0]
    # 桌子无碰撞（contype=conaffinity=0），仅用于视觉显示
    table_geom.contype = 0
    table_geom.conaffinity = 0

    # 机器人挂载 frame（桌面 +X 短边正中央，quat 绕 Z 轴 180° 使机器人朝 -X 即朝桌内）
    robot_frame = scene.worldbody.add_frame()
    robot_frame.name = "robot_mount"
    robot_frame.pos = [1.015, 0, 0.891]
    # MuJoCo 四元数格式 (w,x,y,z)：w=0, z=1 表示绕 Z 轴旋转 180°，机器人朝向 -X（朝内）
    robot_frame.quat = [0, 0, 0, 1]

    # 附加 Franka Panda，兼容 MuJoCo 3.2.x（frame.attach）和 3.5.x+（scene.attach）
    from packaging.version import Version as _V
    if _V(mujoco.__version__) >= _V("3.3"):
        # MuJoCo 3.3+ 正确 API：scene.attach(child, prefix=, frame=)
        scene.attach(panda_spec, prefix="panda_", suffix="", frame=robot_frame)
    else:
        # MuJoCo 3.2.x：frame.attach() 存在但不将 frame.pos 写入 XML
        # 用 XML 补丁确保 panda_link0 获得正确世界坐标
        robot_frame.attach(panda_spec, "panda_", "")

    # 编译
    from packaging.version import Version as _V
    if _V(mujoco.__version__) >= _V("3.3"):
        # 3.5.x+：frame.pos 已正确写入 panda_link0，直接编译
        model = scene.compile()
    else:
        # 3.2.x 补丁：frame.attach 不保留 pos，需通过 XML 中转修正
        import re as _re, tempfile as _tmp
        model_tmp = scene.compile()
        xml_str = scene.to_xml()
        # 将 panda_link0 的 pos/quat 注入 XML
        xml_str = xml_str.replace(
            '<body name="panda_link0" childclass="panda_panda">',
            '<body name="panda_link0" childclass="panda_panda" pos="1.015 0 0.891" quat="0 0 0 1">',
            1,
        )
        # 修正 meshdir 使 panda 网格可被找到
        panda_assets = str(PANDA_XML.parent / "assets")
        table_stl_abs = str(TABLE_STL)
        xml_str = _re.sub(r'meshdir="[^"]*"', f'meshdir="{panda_assets}"', xml_str)
        xml_str = _re.sub(
            r'(<mesh name="vention_table"[^>]*file=")[^"]*(")',
            rf'\g<1>{table_stl_abs}\g<2>',
            xml_str,
        )
        model = mujoco.MjModel.from_xml_string(xml_str)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    return model, data


def main() -> None:
    # 文件存在性检查
    missing = []
    if not TABLE_STL.exists():
        missing.append(f"  桌子 STL 不存在: {TABLE_STL}")
    if not PANDA_XML.exists():
        missing.append(f"  Franka XML 不存在: {PANDA_XML}")
    if missing:
        print("错误：以下文件缺失，请检查路径：", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)

    print("正在组合场景（桌子 + Franka Panda）...")
    model, data = build_scene()

    print(
        f"编译成功: nq={model.nq}, nv={model.nv}, nbody={model.nbody}, "
        f"nmesh={model.nmesh}, ngeom={model.ngeom}"
    )
    print("正在打开 MuJoCo viewer（关闭窗口后脚本退出）...")

    # 阻塞式 viewer，用户关闭窗口时返回
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
