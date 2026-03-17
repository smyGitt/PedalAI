import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

from bilstm import BiLSTMPedalRegressor

CHECKPOINT_PATH = 'pedal_bilstm_checkpoint.pt'
ONNX_PATH = 'pedal_bilstm.onnx'


class BiLSTMWithSigmoid(nn.Module):
    """
    Inference wrapper that appends sigmoid to the raw logit output of
    BiLSTMPedalRegressor. BCEWithLogitsLoss expects raw logits during
    training, so sigmoid is kept out of the base model. This wrapper
    produces pedal probabilities in [0.0, 1.0] for ONNX deployment.
    """
    def __init__(self, model: BiLSTMPedalRegressor):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def export_and_validate(checkpoint_path=CHECKPOINT_PATH, onnx_path=ONNX_PATH):
    device = torch.device('cpu')

    base_model = BiLSTMPedalRegressor()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Sanity-check for NaN weights from a diverged training run.
    nan_params = [(n, p) for n, p in base_model.named_parameters() if not torch.isfinite(p).all()]
    if nan_params:
        raise ValueError(f"Checkpoint contains non-finite weights in: {[n for n, _ in nan_params]}")

    # Wrap with sigmoid so the ONNX output is a pedal probability in [0, 1].
    # Apply threshold >= 0.5 at inference to obtain binary pedal on/off.
    export_model = BiLSTMWithSigmoid(base_model)
    export_model.eval()

    # Dummy input: (batch=1, sequence=256, input_size=140) — 128 piano roll + 12 chroma
    dummy_input = torch.randn(1, 256, 140)

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        input_names=['piano_roll'],
        output_names=['pedal_probability'],
        dynamic_axes={
            'piano_roll':       {0: 'batch_size', 1: 'sequence_length'},
            'pedal_probability': {0: 'batch_size', 1: 'sequence_length'},
        },
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to: {onnx_path}")

    # --- Structural validation ---
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX structural check passed.")

    # --- Numerical validation ---
    # Sequence length 512 != dummy 256 confirms dynamic axes are correctly registered.
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    test_input = torch.randn(1, 512, 140)

    with torch.no_grad():
        torch_out = export_model(test_input).numpy()

    ort_out = ort_session.run(
        None,
        {'piano_roll': test_input.numpy()}
    )[0]

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"Max absolute difference (PyTorch vs ONNX Runtime): {max_diff:.6e}")

    if max_diff < 1e-5:
        print("Numerical validation PASSED.")
    else:
        print(f"WARNING: Numerical difference {max_diff:.6e} exceeds threshold 1e-5.")

    return max_diff


if __name__ == '__main__':
    export_and_validate()
