# run_owg_ui.py
from owg_robot.ui import RobotEnvUI

if __name__ == "__main__":
    # 这里换成你 config 目录里实际存在的 yaml 文件
    cfg_path = "./config/pyb/env.yaml"
    ui = RobotEnvUI(cfg_path)
    ui.run()
