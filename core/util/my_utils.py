
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from datetime import datetime


def get_cur_time_string():
    return timestamp2string_more(time.time())

def timestamp2string(timestmap):
    struct_time = time.localtime(timestmap)
    str = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)
    return str

def timestamp2string_more(timestamp):
    # 将浮点型时间戳转换为 datetime 对象
    dt = datetime.fromtimestamp(timestamp)
    
    # 格式化 datetime 对象为字符串，包括毫秒
    # %f 代表微秒，取前3位得到毫秒
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def draw_traj(traj, text=None, output_dir="tmp"):
    assert isinstance(traj, np.ndarray), "变量不是ndarray类型"
    fig,ax = plt.subplots(figsize=(10,10),dpi=200)
    ax.axis('equal')

    ax.plot(traj[:,0],traj[:,1])
    ax.text(traj[-1,0],traj[-1,1], "e", fontsize=12)
    ax.text(traj[0,0],traj[0,1], "s", fontsize=12)

    if text != None:
        mean_xy = traj.mean(0)
        ax.text(mean_xy[0],mean_xy[1], text)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"mk dir {output_dir}")
    fig_save_path = output_dir / f"draw_traj{get_cur_time_string()}.jpg"
    fig.savefig(fig_save_path)
    print(f"draw candiadate_points at {fig_save_path.resolve()}")



            