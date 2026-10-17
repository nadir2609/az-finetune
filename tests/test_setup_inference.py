from setup_inference import CPU_INDEX, GPU_INDEX, torch_plan


def test_fresh_environment_gets_the_cpu_build():
    assert torch_plan(None, gpu=False, platform="win32") == (False, CPU_INDEX)


def test_macos_installs_from_the_default_index():
    assert torch_plan(None, gpu=False, platform="darwin") == (False, None)


def test_an_installed_torch_is_left_alone_without_the_gpu_flag():
    assert torch_plan("2.11.0+cpu", gpu=False, platform="win32") is None
    assert torch_plan("2.11.0+cu128", gpu=False, platform="win32") is None


def test_gpu_flag_installs_the_cuda_build():
    assert torch_plan(None, gpu=True, platform="win32") == (False, GPU_INDEX)


def test_gpu_flag_replaces_a_cpu_build():
    assert torch_plan("2.14.1+cpu", gpu=True, platform="win32") == (True, GPU_INDEX)


def test_gpu_flag_keeps_an_installed_cuda_build():
    assert torch_plan("2.11.0+cu128", gpu=True, platform="win32") is None
