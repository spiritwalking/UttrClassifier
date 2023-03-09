# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。

from tqdm import tqdm
import time

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。

    pbar = tqdm(range(10))
    for i in pbar:
        time.sleep(0.5)
        pbar.set_description(f"Processing item {i}")


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
