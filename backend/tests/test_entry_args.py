"""入口类构造参数（entry_args）补参通道（2026-10-05，scGPT 补形状失败暴露）。

补形状/两步验证都要**实例化入口类**；当构造参数来自运行期配置或外部数据时（scGPT 的
`TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, vocab=…)`，其中 `vocab[pad_token]`
必须非空），`_model_loader` 的固定猜测列表必然失败——由用户经 IR 的 `entry_args` 给出最小参数。
本文件锁：IR 写入/清除、哈希影响（改它要让旧验证 stale）、脚本优先使用它。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.services import decompose_service, project_manager
from app.services.ir_schema import ir_hash


def _base_ir(project_id: str) -> dict:
    return {
        "schema_version": "1.0", "project_id": project_id,
        "source_file": "model.py", "entry_class": "Net", "task_type": "classification",
        "input_spec": {"shape": [1, 8], "dtype": "float32"}, "root_id": "net",
        "nodes": [{"id": "net", "kind": "module", "class_name": "Net", "parent_id": None,
                   "module_path": ""}],
        "edges": [],
    }


@pytest.fixture()
def ir_env(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    pid = project_manager.create_project("original", source="t")
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    (ws / "reports").mkdir(parents=True, exist_ok=True)
    (ws / "reports" / "ir.json").write_text(json.dumps(_base_ir(pid)), encoding="utf-8")
    return pid


# ---------------------------------------------------------------- 哈希影响

def test_entry_args_in_hash_only_when_present():
    base = _base_ir("p")
    h0 = ir_hash(base)
    assert ir_hash({**base, "entry_args": {"ntoken": 10}}) != h0     # 改了 → 旧验证 stale
    assert ir_hash({**base, "entry_args": {}}) == h0                 # 空 = 不设（不改既有 IR 的哈希）


# ---------------------------------------------------------------- IR 写入/清除

def test_update_entry_args_writes_and_clears(ir_env):
    pid = ir_env
    decompose_service.update_entry_args(pid, {"ntoken": 1000, "vocab": {"<pad>": 0}})
    assert decompose_service.read_ir(pid)["entry_args"] == {"ntoken": 1000, "vocab": {"<pad>": 0}}

    decompose_service.update_entry_args(pid, {})                     # 传空 = 清除
    assert "entry_args" not in decompose_service.read_ir(pid)


def test_update_input_spec_extra_writes_clears_and_validates(ir_env):
    """多输入模型：input_spec.extra 按序补额外入参；传 [] 清除；非正整数 shape 被拒。"""
    pid = ir_env
    spec = decompose_service.update_input_spec(
        pid, [1, 1200], "int64", [{"shape": [1, 1200], "dtype": "float32"}, {"shape": [1, 1200], "dtype": "bool"}])
    assert spec["extra"] == [{"shape": [1, 1200], "dtype": "float32"}, {"shape": [1, 1200], "dtype": "bool"}]

    spec = decompose_service.update_input_spec(pid, [1, 1200], "int64", [])   # 传空 = 清除
    assert "extra" not in spec
    assert "extra" not in decompose_service.read_ir(pid)["input_spec"]

    with pytest.raises(ValueError, match="extra\\[0\\].shape"):
        decompose_service.update_input_spec(pid, [1, 1200], None, [{"shape": [1, 0]}])


def test_update_input_spec_forward_kwargs_writes_clears_and_validates(ir_env):
    """forward 关键字参数（分支开关）：写入/清除；键须为标识符。"""
    pid = ir_env
    spec = decompose_service.update_input_spec(pid, [1, 1200], "int64", None, {"CLS": True, "MVC": True})
    assert spec["forward_kwargs"] == {"CLS": True, "MVC": True}
    assert decompose_service.read_ir(pid)["input_spec"]["forward_kwargs"] == {"CLS": True, "MVC": True}

    spec = decompose_service.update_input_spec(pid, [1, 1200], "int64", None, {})   # 空 = 清除
    assert "forward_kwargs" not in spec

    with pytest.raises(ValueError, match="标识符"):
        decompose_service.update_input_spec(pid, [1, 1200], None, None, {"not-an-ident": 1})


def test_update_input_spec_inputs_validates_node_ids(ir_env):
    """多输入声明：按调用顺序列节点 id；不存在的 id 拒绝（免得生成出对不上的 forward）。"""
    pid = ir_env
    spec = decompose_service.update_input_spec(pid, [1, 8], None, None, None, ["net"])
    assert spec["inputs"] == ["net"]
    assert decompose_service.read_ir(pid)["input_spec"]["inputs"] == ["net"]

    spec = decompose_service.update_input_spec(pid, [1, 8], None, None, None, [])   # 空 = 清除
    assert "inputs" not in spec

    with pytest.raises(ValueError, match="不存在于 IR"):
        decompose_service.update_input_spec(pid, [1, 8], None, None, None, ["nope"])


def test_call_kwargs_reads_forward_kwargs():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import call_kwargs

    assert call_kwargs({"forward_kwargs": {"CLS": True}}) == {"CLS": True}
    assert call_kwargs({}) == {}
    assert call_kwargs({"forward_kwargs": "not-a-dict"}) == {}


def test_make_extra_inputs_builds_tensors():
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import make_extra_inputs

    extras = make_extra_inputs({"extra": [{"shape": [1, 4], "dtype": "float32"},
                                          {"shape": [1, 4], "dtype": "bool"}]})
    assert len(extras) == 2
    assert extras[0].shape == (1, 4) and extras[0].dtype == torch.float32
    assert extras[1].dtype == torch.bool
    assert make_extra_inputs({}) == []              # 无 extra → 空（单输入模型不受影响）


def test_update_entry_args_rejects_non_dict_and_bad_keys(ir_env):
    pid = ir_env
    with pytest.raises(ValueError, match="JSON 对象"):
        decompose_service.update_entry_args(pid, ["not", "a", "dict"])
    with pytest.raises(ValueError, match="标识符"):
        decompose_service.update_entry_args(pid, {"not-an-ident": 1})


def test_entry_args_endpoint(app_client, monkeypatch, tmp_path):
    monkeypatch.setattr(project_manager, "PROJECTS_DIR", tmp_path / "projects")
    pid = project_manager.create_project("original", source="t")
    ws = Path(project_manager.get_project(pid)["workspace_path"])
    (ws / "reports").mkdir(parents=True, exist_ok=True)
    (ws / "reports" / "ir.json").write_text(json.dumps(_base_ir(pid)), encoding="utf-8")

    r = app_client.put(f"/api/projects/{pid}/ir/entry_args",
                       json={"entry_args": {"ntoken": 1000}})
    assert r.status_code == 200 and r.json() == {"ntoken": 1000}
    assert app_client.get(f"/api/projects/{pid}/ir").json()["ir"]["entry_args"] == {"ntoken": 1000}


# ---------------------------------------------------------------- 脚本侧：优先用 entry_args

class _NeedsArgs:
    def __init__(self, ntoken, vocab):        # 必填位置参数（固定猜测列表必然失败）
        self.ntoken, self.vocab = ntoken, vocab


def test_make_dummy_input_handles_integer_dtype():
    """`torch.randn` 不支持整型 dtype（token id 模型会崩）→ 整型用全 0（必在合法索引范围内）。"""
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import make_dummy_input

    f = make_dummy_input([2, 3], torch.float32)
    assert f.shape == (2, 3) and f.dtype == torch.float32

    i = make_dummy_input([2, 3], torch.int64)
    assert i.dtype == torch.int64 and int(i.max()) == 0 and int(i.min()) == 0


def test_redecompose_carries_over_user_specs():
    """重拆解不该清空用户补参（entry_args / extra / forward_kwargs / 用户改过的 shape）。"""
    prev = {
        "entry_args": {"ntoken": 1000, "vocab": {"<pad>": 0}},
        "input_spec": {"shape": [1, 1200], "dtype": "int64", "user_edited": True,
                       "extra": [{"shape": [1, 1200], "dtype": "float32"}],
                       "forward_kwargs": {"CLS": True}},
    }
    fresh = {"input_spec": {"shape": [1, 500], "dtype": "float32"}}   # agent 新产出
    decompose_service._carry_over_user_specs(prev, fresh)

    assert fresh["entry_args"] == {"ntoken": 1000, "vocab": {"<pad>": 0}}
    assert fresh["input_spec"]["extra"] == [{"shape": [1, 1200], "dtype": "float32"}]
    assert fresh["input_spec"]["forward_kwargs"] == {"CLS": True}
    assert fresh["input_spec"]["shape"] == [1, 1200] and fresh["input_spec"]["dtype"] == "int64"

    # 没有上一版（首次拆解）时不动新 IR
    ir = {"input_spec": {"shape": [1, 3]}}
    decompose_service._carry_over_user_specs(None, ir)
    assert ir == {"input_spec": {"shape": [1, 3]}}


def test_update_input_spec_marks_user_edited(ir_env):
    spec = decompose_service.update_input_spec(ir_env, [1, 8], "float32")
    assert spec["user_edited"] is True


def test_accepted_kwargs_filters_by_signature():
    """forward_kwargs 只给签名接受的模型：再生成模型的 forward 由 IR 生成，没有 CLS/MVC 这些开关。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import accepted_kwargs

    def orig_forward(self, src, values, CLS=False, MVC=False):
        ...

    def regen_forward(self, x):
        ...

    assert accepted_kwargs(orig_forward, {"CLS": True, "MVC": True}) == {"CLS": True, "MVC": True}
    assert accepted_kwargs(regen_forward, {"CLS": True, "MVC": True}) == {}

    def with_var_kwargs(self, x, **kw):
        ...

    assert accepted_kwargs(with_var_kwargs, {"CLS": True}) == {"CLS": True}   # **kwargs → 全收
    assert accepted_kwargs(orig_forward, {}) == {}


def test_accepted_positional_truncates_by_arity():
    """再生成模型只声明它消费的输入 → 按签名截断，否则 `takes 2 positional arguments but 4 were given`。

    真实调用传的是**绑定方法** `model.forward`（签名里不含 self）——测试也照此构造。
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import accepted_positional

    class Model:
        def single(self, x):
            ...

        def multi(self, src, values, mask):
            ...

        def varargs(self, *args):
            ...

    m = Model()
    ins = (1, 2, 3)
    assert accepted_positional(m.single, ins) == (1,)
    assert accepted_positional(m.multi, ins) == (1, 2, 3)
    assert accepted_positional(m.varargs, ins) == (1, 2, 3)


def test_first_tensor_unwraps_nested_outputs():
    """dict/tuple 输出取第一个张量：否则返回 Mapping/Tuple 的模型节点永远缺输出形状。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import first_tensor

    import torch

    a, b = torch.zeros(1, 4), torch.zeros(2, 5)
    assert first_tensor({"pred": a, "extra": b}) is a
    assert first_tensor((b, a)) is b
    assert first_tensor([{"x": None}, {"y": a}]) is a
    assert first_tensor("nope") is None
    assert first_tensor({"k": 1.0}) is None


def test_shape_of_swallows_tensors_without_shape():
    """NestedTensor（MHA 快速路径）不支持 `.shape` → 返回 None，不让 hook 抛错中断追踪。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import _shape_of

    import torch

    class Weird:
        @property
        def shape(self):
            raise RuntimeError("NestedTensorImpl doesn't support sizes")

    assert _shape_of(Weird()) is None
    assert _shape_of(torch.zeros(2, 3)) == [2, 3]


def test_instantiate_prefers_entry_args():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import instantiate

    model = instantiate(_NeedsArgs, {"ntoken": 42, "vocab": {"<pad>": 0}})
    assert model.ntoken == 42 and model.vocab == {"<pad>": 0}

    with pytest.raises(RuntimeError, match="无法实例化"):
        instantiate(_NeedsArgs)               # 不给 entry_args → 猜测全失败，如实报错


# --------------------------- checkpoint hparams 兜底（Lightning 系仓库） ---------------------------


class _NeedsStructuredArgs:
    """模拟 boltz 这类：一二十个结构化必填参数，代码里没有直接构造调用。"""

    def __init__(self, atom_s, atom_z, num_bins, msa_args):
        self.atom_s, self.atom_z = atom_s, atom_z
        self.num_bins, self.msa_args = num_bins, msa_args


def _save_fake_ckpt(path: Path, hparams: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}, "hyper_parameters": hparams}, str(path))


def test_instantiate_falls_back_to_checkpoint_hparams(tmp_path, monkeypatch):
    """猜测全失败时，从 checkpoint 的 hyper_parameters 拿构造参数（Lightning 系唯一可靠来源）。

    背景：boltz 的 `Boltz2.__init__` 要 13 个必填结构化参数，仓库里只有
    `load_from_checkpoint`（参数存在 ckpt 的 hparams 里）→ agent 猜不出 → 追踪起不来。
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import instantiate

    ws = tmp_path / "ws"
    (ws / "source").mkdir(parents=True)
    hparams = {"atom_s": 128, "atom_z": 128, "num_bins": 64, "msa_args": {"a": 1}}
    _save_fake_ckpt(ws / "data" / "model.ckpt", hparams)

    model = instantiate(_NeedsStructuredArgs, None, str(ws / "source"))
    assert model.num_bins == 64 and model.msa_args == {"a": 1}


def test_instantiate_checkpoint_is_last_resort(tmp_path, monkeypatch):
    """entry_args 能给就不读权重（读 ckpt 很贵），且明确指定的 checkpoint 优先。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import instantiate

    ws = tmp_path / "ws"
    (ws / "source").mkdir(parents=True)
    _save_fake_ckpt(ws / "data" / "wrong.ckpt",
                    {"atom_s": 1, "atom_z": 1, "num_bins": 1, "msa_args": {}})
    right = tmp_path / "right.ckpt"
    _save_fake_ckpt(right, {"atom_s": 9, "atom_z": 9, "num_bins": 9, "msa_args": {"b": 2}})
    monkeypatch.setenv("DECOMPOSE_CHECKPOINT", str(right))

    # entry_args 足够 → 不去读权重
    got = instantiate(_NeedsStructuredArgs, {"atom_s": 3, "atom_z": 3, "num_bins": 3, "msa_args": {}},
                      str(ws / "source"))
    assert got.num_bins == 3

    # 没有 entry_args → 读显式指定的那个
    got2 = instantiate(_NeedsStructuredArgs, None, str(ws / "source"))
    assert got2.num_bins == 9


def test_instantiate_without_checkpoint_still_reports_signature(tmp_path):
    """没有 checkpoint 时行为不变：如实抛「无法实例化」并列出尝试过的候选。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import instantiate

    ws = tmp_path / "ws"
    (ws / "source").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="无法实例化"):
        instantiate(_NeedsStructuredArgs, None, str(ws / "source"))


# --------------------------- 造输入约定 make_inputs.py ---------------------------


def test_build_inputs_prefers_project_make_inputs(tmp_path):
    """有项目约定脚本就用它（boltz 这种要 `feats` 字典的模型靠它）；没有就退回 dummy 张量。

    背景：真实仓库的 forward 常不吃「一个张量」——boltz 要 feats 特征字典（键多、维度各异，
    由 featurizer 造）。平台按 shape+dtype 喂 dummy 张量必然 `IndexError`（实测踩到），
    追踪/验证全卡在第一步。
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import build_inputs

    ws = tmp_path / "ws"
    src = ws / "source"
    src.mkdir(parents=True)
    spec = {"shape": [1, 4], "dtype": "float32"}

    # ① 没有约定脚本 → 退回 shape+dtype 的 dummy 张量（旧行为不变）
    xs = build_inputs(object(), spec, str(src))
    assert len(xs) == 1 and tuple(xs[0].shape) == (1, 4)

    # ② 有约定脚本 → 用脚本产物（这里返回一个字典实参）
    (ws / "reports").mkdir()
    (ws / "reports" / "make_inputs.py").write_text(
        "def build_inputs(model):\n"
        "    import torch\n"
        "    return ({'feats': torch.zeros(2, 3)},)\n",
        encoding="utf-8",
    )
    xs2 = build_inputs(object(), spec, str(src))
    assert list(xs2[0]["feats"].shape) == [2, 3]


def test_build_inputs_falls_back_when_script_broken(tmp_path):
    """约定脚本写坏（导入报错/没定义 build_inputs）→ 不阻断，退回 dummy 张量。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import build_inputs

    ws = tmp_path / "ws"
    src = ws / "source"
    (ws / "reports").mkdir(parents=True)
    src.mkdir()
    (ws / "reports" / "make_inputs.py").write_text("import not_a_real_module\n", encoding="utf-8")

    xs = build_inputs(object(), {"shape": [1, 4], "dtype": "float32"}, str(src))
    assert tuple(xs[0].shape) == (1, 4)


def test_instantiate_prefers_project_make_model(tmp_path):
    """有 `make_model.py` 就优先用它——仓库的 `load_from_checkpoint(**覆盖)` 里有 ckpt 没有的参数。

    实测 boltz：ckpt 的 `pairformer_args` 没有 `v2`，而当前代码的 forward 无条件用 v2 才有的
    `k_in` → 光靠 ckpt 的 hparams 构造出来的模型跑不了。
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import instantiate

    class _OneArg:
        def __init__(self, n):
            self.n = n

    ws = tmp_path / "ws"
    src = ws / "source"
    src.mkdir(parents=True)
    (ws / "reports").mkdir()
    (ws / "reports" / "make_model.py").write_text(
        "def build_model(entry_class, checkpoint=None):\n"
        "    return entry_class(42)          # 模拟「照仓库的加载方式」构造\n",
        encoding="utf-8",
    )
    # 即使给了 entry_args，也以项目构造脚本为准（它最权威）
    assert instantiate(_OneArg, {"n": 1}, str(src)).n == 42


def test_instantiate_make_model_returning_none_falls_back(tmp_path):
    """约定脚本返回 None（如没权重）→ 回退其它构造方式，不误当成成功。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from _model_loader import instantiate

    ws = tmp_path / "ws"
    src = ws / "source"
    src.mkdir(parents=True)
    (ws / "reports").mkdir()
    (ws / "reports" / "make_model.py").write_text(
        "def build_model(entry_class, checkpoint=None):\n"
        "    return None\n", encoding="utf-8")
    assert instantiate(_NeedsArgs, {"ntoken": 7, "vocab": {}}, str(src)).ntoken == 7
