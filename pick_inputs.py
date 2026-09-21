"""Native local file picker, separate process to keep Tk on its main thread."""
import json
import sys
import tkinter as tk
from tkinter import filedialog

sys.stdout.reconfigure(encoding='utf-8')
window=tk.Tk();window.withdraw();window.attributes('-topmost',True)
try:
    kind=sys.argv[1]
    if kind=='mesh':
        path=filedialog.askopenfilename(parent=window,title='选择人脸模型 OBJ',filetypes=[('OBJ model','*.obj')])
        result={'path':path}
    elif kind=='references':
        paths=filedialog.askopenfilenames(parent=window,title='选择参考图（可多选）',filetypes=[('Images','*.png *.jpg *.jpeg *.webp'),('All files','*.*')])
        result={'paths':list(paths)}
    elif kind=='output':result={'path':filedialog.askdirectory(parent=window,title='选择输出目录')}
    else:raise ValueError('Unknown picker type')
    print(json.dumps(result,ensure_ascii=False))
finally:window.destroy()
