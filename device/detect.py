import numpy as np
import sounddevice as sd
import json, csv, os, time
from datetime import datetime
from threading import Thread, Timer
import tkinter as tk

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ── Config ───────────────────────────────────────────────────
SAMPLE_RATE  = 16000
DURATION     = 3
LOG_INTERVAL = 15
THRESHOLD    = 0.75
MODEL_DIR    = '/root/flood_detector'
LOG_PATH     = f'{MODEL_DIR}/log.csv'

# ── Tkinter root ──────────────────────────────────────────────
root = tk.Tk()
root.title('Flood Detector')
root.geometry('240x320')
root.configure(bg='white')
root.resizable(False, False)

# ── Loading screen ────────────────────────────────────────────
loading_lbl = tk.Label(root, text='Loading YAMNet...',
                       font=('Arial', 11), bg='white')
loading_lbl.place(x=120, y=160, anchor='center')
root.update()

# ── Load models ───────────────────────────────────────────────
yamnet_interp = Interpreter(f'{MODEL_DIR}/yamnet_fixed.tflite')
yamnet_interp.allocate_tensors()
yamnet_inp = yamnet_interp.get_input_details()
yamnet_out = yamnet_interp.get_output_details()

loading_lbl.config(text='Loading classifier...')
root.update()

clf_interp = Interpreter(f'{MODEL_DIR}/flood_classifier_only.tflite')
clf_interp.allocate_tensors()
clf_inp = clf_interp.get_input_details()
clf_out = clf_interp.get_output_details()

with open(f'{MODEL_DIR}/class_labels.json') as f:
    CLASSES = json.load(f)

# ── Log file ──────────────────────────────────────────────────
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow([
            'timestamp', 'flood_water_%', 'rain_%', 'ambient_dry_%', 'prediction'
        ])

def log_event(probs, pred_cls):
    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f'{probs[0]:.1%}',
            f'{probs[1]:.1%}',
            f'{probs[2]:.1%}',
            pred_cls,
        ])

# ── Inference ─────────────────────────────────────────────────
def get_embedding(audio):
    audio = (audio / (np.max(np.abs(audio)) + 1e-6)).astype(np.float32)
    audio = audio[:48000] if len(audio) >= 48000 \
            else np.pad(audio, (0, 48000 - len(audio)))
    yamnet_interp.set_tensor(yamnet_inp[0]['index'], audio.reshape(1, 48000))
    yamnet_interp.invoke()
    return yamnet_interp.get_tensor(yamnet_out[0]['index']).copy()

def predict(audio):
    emb = get_embedding(audio)
    clf_interp.set_tensor(clf_inp[0]['index'], emb)
    clf_interp.invoke()
    return clf_interp.get_tensor(clf_out[0]['index'])[0]

# ── Backlight ─────────────────────────────────────────────────
import subprocess

def set_backlight(on: bool):
    try:
        if on:
            subprocess.run(['xset', 's', 'reset'])
        else:
            subprocess.run(['xset', 's', 'activate'])
    except Exception as e:
        print(f'Backlight error: {e}')

# ── State ─────────────────────────────────────────────────────
screen_refs      = [None]
state            = {'running': True, 'screen_on': True, 'last_log': 0}
_long_press_timer = [None]

# ── Screen OFF / ON ───────────────────────────────────────────
def go_screen_off():
    set_backlight(False)
    state['screen_on'] = False
    screen_refs[0] = None
    for w in root.winfo_children():
        w.destroy()

    canvas = tk.Canvas(root, width=240, height=320, bg='black',
                       highlightthickness=0)
    canvas.pack(fill='both', expand=True)

    def on_press(e):
        t = Timer(0.8, lambda: root.after(0, go_screen_on))
        _long_press_timer[0] = t
        t.start()

    def on_release(e):
        if _long_press_timer[0]:
            _long_press_timer[0].cancel()
            _long_press_timer[0] = None

    canvas.bind('<ButtonPress-1>',   on_press)
    canvas.bind('<ButtonRelease-1>', on_release)

def go_screen_on():
    set_backlight(True)
    if _long_press_timer[0]:
        _long_press_timer[0] = None
    state['screen_on'] = True
    rebuild_full_ui()

# ── Active UI ─────────────────────────────────────────────────
def build_active_ui():
    for w in root.winfo_children():
        w.destroy()
    root.configure(bg='white')

    tk.Label(root, text='Flood Detector', font=('Arial', 14, 'bold'),
             bg='white', fg='blue').place(x=120, y=10, anchor='center')

    status_lbl = tk.Label(root, text='Listening...', font=('Arial', 11),
                          bg='white', fg='black')
    status_lbl.place(x=120, y=46, anchor='center')

    pred_lbl = tk.Label(root, text='', font=('Arial', 13, 'bold'),
                        bg='white', fg='black')
    pred_lbl.place(x=120, y=65, anchor='center')

    conf_lbl = tk.Label(root, text='', font=('Arial', 11),
                        bg='white', fg='grey')
    conf_lbl.place(x=120, y=83, anchor='center')

    alert_lbl = tk.Label(root, text='', font=('Arial', 12, 'bold'),
                         bg='white', fg='red')
    alert_lbl.place(x=120, y=102, anchor='center')

    colors = ['blue', 'green', 'orange']
    labels = ['flood', 'rain', 'dry']
    bar_canvases, bar_lbls = [], []

    for i, (lbl, col) in enumerate(zip(labels, colors)):
        y = 138 + i * 34
        tk.Label(root, text=lbl, font=('Arial', 10),
                 bg='white').place(x=8, y=y)
        c = tk.Canvas(root, width=125, height=17, bg='white',
                      highlightthickness=0)
        c.place(x=52, y=y)
        bar_canvases.append((c, col))
        bl = tk.Label(root, text='0%', font=('Arial', 10), bg='white')
        bl.place(x=185, y=y)
        bar_lbls.append(bl)

    return status_lbl, pred_lbl, conf_lbl, alert_lbl, bar_lbls, bar_canvases

def update_screen(probs, pred_cls, conf):
    if not screen_refs[0] or not state['screen_on']:
        return
    status_lbl, pred_lbl, conf_lbl, alert_lbl, bar_lbls, bar_canvases = screen_refs[0]
    def _update():
        try:
            pred_lbl.config(text=pred_cls)
            conf_lbl.config(text=f'{conf:.0%} confidence')
            status_lbl.config(text=datetime.now().strftime('%H:%M:%S'))
            for i, prob in enumerate(probs):
                c, col = bar_canvases[i]
                c.delete('all')
                c.create_rectangle(0, 0, max(int(prob * 125), 1), 17,
                                   fill=col, outline='')
                bar_lbls[i].config(text=f'{prob:.0%}')
            alert_lbl.config(
                text='FLOOD DETECTED!' if pred_cls == 'flood_water'
                and conf >= THRESHOLD else ''
            )
        except Exception:
            pass
    root.after(0, _update)

def rebuild_full_ui():
    refs = build_active_ui()
    screen_refs[0] = refs

    def start_stop():
        state['running'] = not state['running']
        start_btn.config(text='STOP' if state['running'] else 'START',
                         bg='salmon' if state['running'] else 'lightgrey')
        screen_refs[0][0].config(
            text='Listening...' if state['running'] else 'Stopped'
        )

    start_btn = tk.Button(root, text='STOP', command=start_stop,
                          width=20, bg='salmon')
    start_btn.place(x=120, y=255, anchor='center')

    tk.Button(root, text='Screen OFF', command=go_screen_off,
              width=20, bg='lightgrey').place(x=120, y=290, anchor='center')

rebuild_full_ui()

# ── Inference loop ────────────────────────────────────────────
# ── Inference loop ────────────────────────────────────────────
N = int(SAMPLE_RATE * DURATION)

def inference_loop():
    buffer = np.zeros(N, dtype='float32')
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='float32', blocksize=N) as stream:
        while True:
            try:
                if state['running']:
                    chunk, _ = stream.read(N)
                    buffer[:] = chunk.flatten()

                    probs    = predict(buffer)
                    pred_idx = np.argmax(probs)
                    pred_cls = CLASSES[pred_idx]
                    conf     = float(probs[pred_idx])

                    if state['screen_on']:
                        update_screen(probs, pred_cls, conf)

                    now = time.time()
                    if now - state['last_log'] >= LOG_INTERVAL:
                        log_event(probs, pred_cls)
                        state['last_log'] = now
                        print(f'[{datetime.now().strftime("%H:%M:%S")}] '
                              f'{pred_cls} {conf:.0%}')
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f'Inference error: {e}')
                time.sleep(0.5)

# ── Start ─────────────────────────────────────────────────────
Thread(target=inference_loop, daemon=True).start()
print('Flood Detector running')

root.mainloop()