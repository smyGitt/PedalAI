#!/usr/bin/env python3
"""One-time conversion: pedal_bilstm.onnx -> pedal_bilstm.npz
Run this once, then onnx can be uninstalled."""
import os
import numpy as np
import onnx
import onnx.numpy_helper

base = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(base, 'pedal_bilstm.onnx')
npz_path  = os.path.join(base, 'pedal_bilstm.npz')

m = onnx.load(onnx_path)
w = {init.name: onnx.numpy_helper.to_array(init) for init in m.graph.initializer}

np.savez(npz_path,
    lstm1_W = w['onnx::LSTM_352'],
    lstm1_R = w['onnx::LSTM_353'],
    lstm1_B = w['onnx::LSTM_351'],
    lstm2_W = w['onnx::LSTM_395'],
    lstm2_R = w['onnx::LSTM_396'],
    lstm2_B = w['onnx::LSTM_394'],
    linear_W = w['onnx::MatMul_397'],
    linear_B = w['model.regression_head.bias'],
)

print(f"Saved {npz_path}")
print("You can now uninstall onnx:  pip uninstall onnx")
