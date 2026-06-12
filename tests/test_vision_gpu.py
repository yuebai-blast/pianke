import torch

from pic_selecter import vision


def _fake_device(kind: str):
    return type("FakeDevice", (), {"type": kind})()


def test_select_onnx_providers_prefers_cuda(monkeypatch):
    monkeypatch.delenv("PIC_SELECTER_RUNTIME", raising=False)
    monkeypatch.setattr(vision, "_device", lambda: _fake_device("cuda"))
    assert vision._select_onnx_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        0,
    )


def test_select_onnx_providers_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("PIC_SELECTER_RUNTIME", raising=False)
    monkeypatch.setattr(vision, "_device", lambda: _fake_device("cuda"))
    assert vision._select_onnx_providers(["CPUExecutionProvider"]) == (
        ["CPUExecutionProvider"],
        -1,
    )


def test_pyiqa_device_keeps_mps_on_cpu(monkeypatch):
    monkeypatch.delenv("PIC_SELECTER_PYIQA_DEVICE", raising=False)
    monkeypatch.delenv("PIC_SELECTER_RUNTIME", raising=False)
    monkeypatch.setattr(vision, "_device", lambda: _fake_device("mps"))
    assert vision._pyiqa_device() == torch.device("cpu")


def test_runtime_cpu_forces_cpu_onnx(monkeypatch):
    monkeypatch.setenv("PIC_SELECTER_RUNTIME", "cpu")
    assert vision._select_onnx_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == (
        ["CPUExecutionProvider"],
        -1,
    )


def test_runtime_gpu_requires_accelerator(monkeypatch):
    monkeypatch.setenv("PIC_SELECTER_RUNTIME", "gpu")
    monkeypatch.delenv("PIC_SELECTER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    vision._DEVICE = None
    try:
        try:
            vision._device()
        except vision.VisionUnavailable:
            pass
        else:
            raise AssertionError("expected VisionUnavailable")
    finally:
        vision._DEVICE = None


def test_reset_runtime_state_clears_cached_models():
    vision._DEVICE = "sentinel"
    vision._models["demo"] = object()
    vision.reset_runtime_state()
    assert vision._DEVICE is None
    assert vision._models == {}
