"""Small governed ONNX bridge used by PhysioML."""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort


def _shape(value):
    out = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            out.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            out.append(int(dim.dim_value))
        else:
            out.append(None)
    return out


def _dtype(value):
    if value == onnx.TensorProto.FLOAT:
        return "float32"
    return onnx.helper.tensor_dtype_to_string(value).lower()


def versions():
    try:
        import torch

        python_torch = torch.__version__
    except ImportError:
        python_torch = None
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "python_torch": python_torch,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
    }


def inspect_model(path, full_check=True):
    graph = onnx.load(os.fspath(path), load_external_data=False)
    onnx.checker.check_model(graph, full_check=bool(full_check))
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("expected exactly one graph input and one graph output")
    graph_input = graph.graph.input[0]
    graph_output = graph.graph.output[0]
    return {
        "ir_version": int(graph.ir_version),
        "producer_name": graph.producer_name,
        "producer_version": graph.producer_version,
        "opset": int(graph.opset_import[0].version),
        "input": {
            "name": graph_input.name,
            "dtype": _dtype(graph_input.type.tensor_type.elem_type),
            "shape": _shape(graph_input),
        },
        "output": {
            "name": graph_output.name,
            "dtype": _dtype(graph_output.type.tensor_type.elem_type),
            "shape": _shape(graph_output),
        },
    }


def _cpu_session(path):
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
    session = ort.InferenceSession(
        os.fspath(path),
        providers=["CPUExecutionProvider"],
    )
    session.disable_fallback()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX Runtime provider fallback was not disabled")
    return session


def predict_model(path, inputs):
    np_state = np.random.get_state()
    try:
        array = np.ascontiguousarray(inputs, dtype=np.float32)
        session = _cpu_session(path)
        output = session.run(["output"], {"signal": array})[0]
        return {
            "output": np.asarray(output, dtype=np.float32),
            "providers": session.get_providers(),
        }
    finally:
        np.random.set_state(np_state)


def export_model(
    torchscript_path,
    onnx_path,
    inputs,
    opset,
    dynamic_batch,
    full_check,
):
    import torch

    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        array = np.ascontiguousarray(inputs, dtype=np.float32)
        module = torch.jit.load(os.fspath(torchscript_path), map_location="cpu")
        module.eval()
        tensor = torch.from_numpy(array)
        dynamic_axes = None
        if bool(dynamic_batch):
            dynamic_axes = {
                "signal": {0: "batch"},
                "output": {0: "batch"},
            }
        torch.onnx.export(
            module,
            tensor,
            os.fspath(onnx_path),
            input_names=["signal"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
        )
        summary = inspect_model(onnx_path, full_check=full_check)
        prediction = predict_model(onnx_path, array)
        return {
            "summary": summary,
            "output": prediction["output"],
            "providers": prediction["providers"],
            "versions": versions(),
        }
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
