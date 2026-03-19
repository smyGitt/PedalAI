#!/usr/bin/env python3
"""One-time conversion: pedal_bilstm.onnx -> pedal_bilstm.npz
Run this once, then onnx can be uninstalled."""
import os
import numpy as np
import onnx
import onnx.numpy_helper

base = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(base, '..', 'pedal_bilstm.onnx')
npz_path  = os.path.join(base, '..', 'pedal_bilstm.npz')

m = onnx.load(onnx_path)
w = {init.name: onnx.numpy_helper.to_array(init) for init in m.graph.initializer}

# Find LSTM and linear weights by shape instead of hardcoded names.
# LSTM W shape: (num_directions, 4*hidden_size, input_size_or_hidden_size)
# LSTM R shape: (num_directions, 4*hidden_size, hidden_size)
# LSTM B shape: (num_directions, 8*hidden_size)
# Linear W shape: (hidden_size*2, 1) via MatMul
# Linear B shape: (1,)

lstm_w = []  # weight matrices (W)
lstm_r = []  # recurrence matrices (R)
lstm_b = []  # biases (B)
linear_w = None
linear_b = None

for name, arr in w.items():
    if arr.ndim == 3 and arr.shape[0] == 2:  # bidirectional LSTM
        if arr.shape[2] == arr.shape[1] // 4:  # R: (2, 4*H, H)
            lstm_r.append((name, arr))
        else:  # W: (2, 4*H, input_size)
            lstm_w.append((name, arr))
    elif arr.ndim == 2 and arr.shape[0] == 2:  # B: (2, 8*H)
        lstm_b.append((name, arr))
    elif arr.ndim == 2 and arr.shape[1] == 1:  # linear W: (H*2, 1)
        linear_w = arr
    elif arr.ndim == 1 and arr.shape[0] == 1:  # linear B: (1,)
        linear_b = arr

# Sort by input size to distinguish layer 1 (input_size=140) from layer 2 (input_size=hidden*2)
lstm_w.sort(key=lambda x: -x[1].shape[2])  # larger input first = layer 1
lstm_r.sort(key=lambda x: x[0])  # consistent ordering
lstm_b.sort(key=lambda x: x[0])

assert len(lstm_w) == 2, f"Expected 2 LSTM W matrices, found {len(lstm_w)}"
assert len(lstm_r) == 2, f"Expected 2 LSTM R matrices, found {len(lstm_r)}"
assert len(lstm_b) == 2, f"Expected 2 LSTM B matrices, found {len(lstm_b)}"
assert linear_w is not None, "Linear weight not found"
assert linear_b is not None, "Linear bias not found"

np.savez(npz_path,
    lstm1_W = lstm_w[0][1],
    lstm1_R = lstm_r[0][1],
    lstm1_B = lstm_b[0][1],
    lstm2_W = lstm_w[1][1],
    lstm2_R = lstm_r[1][1],
    lstm2_B = lstm_b[1][1],
    linear_W = linear_w,
    linear_B = linear_b,
)

print(f"Saved {npz_path}")
print("You can now uninstall onnx:  pip uninstall onnx")
