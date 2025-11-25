"""
PyTorch 兼容性补丁

修复 PyTorch 2.0.0 与 unstructured 库的兼容性问题，采用延迟加载避免
在启动阶段拉起重依赖。
"""

import importlib
import sys


def _get_torch():
    """延迟加载 torch，避免启动阶段的重依赖开销"""
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def patch_pytorch_pytree():
    """
    为旧版本 PyTorch 添加缺失的 _pytree 功能
    修复 'register_pytree_node' 错误
    """
    torch = _get_torch()
    if torch is None:
        return False

    if hasattr(torch.utils, "_pytree"):
        if not hasattr(torch.utils._pytree, "register_pytree_node"):
            def register_pytree_node(cls, flatten_fn, unflatten_fn):
                """虚拟实现，避免 AttributeError"""
                return cls

            torch.utils._pytree.register_pytree_node = register_pytree_node
            print("已应用 PyTorch _pytree 兼容性补丁")
            return True
    else:
        class DummyPytree:
            @staticmethod
            def register_pytree_node(cls, flatten_fn, unflatten_fn):
                return cls

        torch.utils._pytree = DummyPytree()
        print("已创建虚拟 _pytree 模块")
        return True

    return False


def check_pytorch_compatibility():
    """检查 PyTorch 版本兼容性"""
    torch = _get_torch()
    if torch is None:
        return {
            "version": None,
            "major": None,
            "minor": None,
            "has_pytree": False,
            "has_register_pytree_node": False,
            "needs_patch": False,
        }

    pytorch_version = torch.__version__
    major, minor = map(int, pytorch_version.split(".")[:2])

    compatibility_info = {
        "version": pytorch_version,
        "major": major,
        "minor": minor,
        "has_pytree": hasattr(torch.utils, "_pytree"),
        "has_register_pytree_node": False,
        "needs_patch": False,
    }

    if hasattr(torch.utils, "_pytree"):
        compatibility_info["has_register_pytree_node"] = hasattr(
            torch.utils._pytree, "register_pytree_node"
        )

    # PyTorch 2.0.x 需要补丁
    if major == 2 and minor == 0:
        compatibility_info["needs_patch"] = True

    return compatibility_info


def auto_patch():
    """自动检测并应用必要的补丁"""
    compat_info = check_pytorch_compatibility()

    if compat_info["needs_patch"] and not compat_info["has_register_pytree_node"]:
        print(f"检测到 PyTorch {compat_info['version']}，正在应用兼容性补丁...")
        patch_pytorch_pytree()
        return True

    return False
