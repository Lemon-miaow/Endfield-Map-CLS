"""
report.py — 地图分类模型更新报告

用 validation_images 里的固定真实验证帧（地图类 + 无小地图）对比线上模型与候选模型，
画成一张能直接贴进 PR 的 PNG：认对数、正确类平均/最低置信度、低于 Aug11 基线的样本数、
置信度有变化的逐样本对比，以及变化最大的几张原图。

模型可以是 ONNX 文件（同名 .json 提供类别表），也可以是 MaaEnd-AI 的提交号
（取该提交的 map/cls.onnx 与 map/cls.json）。

用法:
    python report.py --online 23ef080 --candidate c674cef
                     [--online-name best35] [--candidate-name best42]
                     [--baseline 3a822e8] [--out model_report.png]
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

DEFAULT_OPTIONS = {
    "baseline": "3a822e8",  # MaaEnd-AI 里的 Aug11 模型，逐样本门槛的基准
    "model_repo": "~/MaaEnd-AI",
    "maaend": "~/MaaEnd",
    "validation": "validation_images",
    "source": "source_images",
    "out": "model_report.png",
}

# 有变化才列进逐样本对比：变化不足 0.5 个百分点且两版都在 95% 以上的样本只计入汇总。
CHANGE_MIN_PT = 0.5
LOW_SCORE_PT = 95.0
GATE_MARGIN_PT = 1.0
THUMBNAIL_COUNT = 6

INK = {
    "surface": "#fcfcfb",
    "tile": "#f9f9f7",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "#e1e0d9",
    "up": "#006300",
    "down": "#d03b3b",
}
ONLINE_COLOR = "#eb6834"
CANDIDATE_COLOR = "#2a78d6"

CJK_FONTS = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

MAP_NAMES = {"map01": "四号谷地", "map02": "武陵"}
# 能量淤积点数据覆盖了武陵除景玉谷以外的全部区域，景玉谷由排除法对上 map02_lv001。
EXTRA_LEVEL_NAMES = {"map02_lv001": "景玉谷"}

# 测试集无小地图帧的文件名词汇；新样本里没收录的词原样显示。
SCENE_WORDS = {
    "photo_mode": "拍照模式", "indoor_crops": "室内", "operator_text": "干员文字", "target": "准星",
    "dark_forest": "暗林", "white_flowers": "白花", "close": "近景", "sky": "天空", "horizon": "地平线",
    "building_shrub": "楼与灌木", "blue_roof": "蓝屋顶", "dark_pillars": "暗色立柱", "tree_trunk": "树干",
    "pipes": "管道", "shell": "外壳",
}

SOURCE_LABELS = (
    ("autocollect_", "自动采集定位帧"),
    ("onerror_", "线上报错帧"),
    ("user_", "用户反馈帧"),
    ("icon_dense", "密集图标帧"),
    ("ultra_dense", "超密图标帧"),
    ("yellow_zone", "黄圈图标帧"),
    ("transparent_", "透明区透出帧"),
    ("zipline_ride", "滑索滑行帧"),
)


@dataclass(frozen=True)
class Model:
    name: str
    md5: str
    classes: list[str]
    net: cv2.dnn.Net

    def scores(self, image: np.ndarray) -> np.ndarray:
        self.net.setInput(cv2.dnn.blobFromImage(image, 1 / 255.0, (128, 128), swapRB=True))
        return self.net.forward().reshape(-1)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    image: np.ndarray


def git_show(repo: Path, spec: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", spec], capture_output=True, check=True
    ).stdout


def load_model(spec: str, name: str | None, model_repo: Path) -> Model:
    """ONNX 文件或 MaaEnd-AI 提交号 → 模型；类别表取同名 .json 或提交里的 cls.json。"""
    path = Path(spec).expanduser()
    if path.is_file():
        onnx = path.read_bytes()
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    else:
        onnx = git_show(model_repo, f"{spec}:map/cls.onnx")
        meta = json.loads(git_show(model_repo, f"{spec}:map/cls.json"))
    classes = meta["classes"]
    if isinstance(classes, str):
        classes = ast.literal_eval(classes)
    return Model(
        name=name or spec,
        md5=hashlib.md5(onnx).hexdigest()[:8],
        classes=list(classes),
        net=cv2.dnn.readNetFromONNX(np.frombuffer(onnx, np.uint8)),
    )


def load_samples(validation_dir: Path) -> list[Sample]:
    return [
        Sample(path, class_dir.name, cv2.imread(str(path)))
        for class_dir in sorted(p for p in validation_dir.iterdir() if p.is_dir())
        for path in sorted(class_dir.glob("*.png"))
    ]


class RegionNamer:
    """类名 → 区域名：Base 格按格心落在哪个区域框，Tier 类直接取名字里的 Lv 编号。"""

    def __init__(self, maaend: Path, source_dir: Path):
        self.level_names = dict(EXTRA_LEVEL_NAMES)
        gems = maaend / "assets/data/EssenceFilter/energy_point_gems.json"
        if gems.exists():
            for point in json.loads(gems.read_text(encoding="utf-8")):
                self.level_names.setdefault(point["levelId"], point["pointName"].split("·")[-1])

        self.layouts = {}
        for map_id in MAP_NAMES:
            layout_path = maaend / f"assets/data/ZmdMap/{map_id}_layout.json"
            base_path = source_dir / f"Map{map_id[-2:]}Base.png"
            if not layout_path.exists() or not base_path.exists():
                continue
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            base_width = cv2.imread(str(base_path), cv2.IMREAD_UNCHANGED).shape[1]
            self.layouts[map_id] = (layout["canvas_width"] / base_width, layout["levels"])

    def level_name(self, map_id: str, level: str) -> str:
        return self.level_names.get(f"{map_id}_{level}", level)

    def base_level(self, map_id: str, row: int, col: int, tile: int = 160) -> str:
        scale, levels = self.layouts[map_id]
        x, y = (col + 0.5) * tile * scale, (row + 0.5) * tile * scale

        def distance(rect: dict) -> float:
            cx, cy = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
            inside = rect["x"] <= x <= rect["x"] + rect["width"] and rect["y"] <= y <= rect["y"] + rect["height"]
            return (0 if inside else 1e9) + np.hypot(x - cx, y - cy)

        return min(levels, key=lambda key: distance(levels[key])).split("_")[-1]

    def __call__(self, label: str) -> str:
        if label == "None":
            return "无小地图"
        base = re.fullmatch(r"Map(\d\d)Base__r(\d+)_c(\d+)", label)
        if base and f"map{base[1]}" in self.layouts:
            map_id = f"map{base[1]}"
            level = self.base_level(map_id, int(base[2]), int(base[3]))
            return f"{MAP_NAMES[map_id]}/{self.level_name(map_id, level)}（r{base[2]}_c{base[3]}）"
        tier = re.fullmatch(r"Map(\d\d)Lv(\d{3})Tier(\d+)", label)
        if tier:
            map_id = f"map{tier[1]}"
            return f"{MAP_NAMES[map_id]}/{self.level_name(map_id, 'lv' + tier[2])}（Tier{tier[3]}）"
        return label


def source_label(path: Path) -> str:
    """文件名 → 样本来源说明，认不出的保留文件名。"""
    stem = path.stem
    issue = re.match(r"issue_(\d+)", stem)
    if issue:
        return f"issue #{issue[1]}"
    if stem.startswith("testset_"):
        # testset_<来源目录>_<编号>_<描述>：只留描述，按词表翻译
        scene = re.sub(r"^(transfer|port_storager)_\d+_", "", stem.removeprefix("testset_"))
        for word, text in sorted(SCENE_WORDS.items(), key=lambda item: -len(item[0])):
            scene = scene.replace(word, text)
        return "测试集·" + scene.replace("_", "")
    for prefix, text in SOURCE_LABELS:
        if stem.startswith(prefix):
            return text
    if re.match(r"\d{6,}-", stem):
        return "issue 附图"
    return "实机帧"


def changed_rows(old: np.ndarray, new: np.ndarray) -> list[int]:
    """有变化或偏低的样本，按候选相对线上的变化从升到降排列。"""
    delta = new - old
    keep = (np.abs(delta) >= CHANGE_MIN_PT) | (np.minimum(old, new) < LOW_SCORE_PT)
    return sorted(np.flatnonzero(keep).tolist(), key=lambda i: -delta[i])


def use_cjk_font() -> None:
    for path in CJK_FONTS:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def delta_text(delta: float, digits: int = 1, lower_is_better: bool = False) -> tuple[str, str]:
    """箭头跟着数值涨跌，颜色跟着好坏。"""
    if abs(delta) < 10 ** -digits / 2:
        return "持平", INK["muted"]
    better = (delta < 0) if lower_is_better else (delta > 0)
    return f"{'▲' if delta > 0 else '▼'} {abs(delta):.{digits}f}", INK["up"] if better else INK["down"]


def draw_tile(fig, rect, title: str, value: str, online: str, delta: tuple[str, str]) -> None:
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.06",
        transform=ax.transAxes, facecolor=INK["tile"], edgecolor=INK["border"], linewidth=1,
    ))
    ax.text(0.08, 0.76, title, fontsize=12, color=INK["secondary"], transform=ax.transAxes)
    ax.text(0.08, 0.36, value, fontsize=26, color=INK["primary"], transform=ax.transAxes)
    ax.text(0.08, 0.14, online, fontsize=11, color=INK["muted"], transform=ax.transAxes)
    ax.text(0.92, 0.14, delta[0], fontsize=11, color=delta[1], ha="right", transform=ax.transAxes)


def render(samples: list[Sample], namer: RegionNamer, models: list[Model | None], scores: dict, out: Path) -> None:
    online, candidate, baseline = models
    labels = [sample.label for sample in samples]
    regions = [namer(label) for label in labels]
    old, new = scores[online.name], scores[candidate.name]
    base = scores[baseline.name] if baseline else None
    rows = changed_rows(old, new)
    thumbs = sorted((i for i in rows if abs(new[i] - old[i]) >= GATE_MARGIN_PT), key=lambda i: -abs(new[i] - old[i]))[:THUMBNAIL_COUNT]

    width, row_h = 12.0, 0.34
    chart_h = max(len(rows), 1) * row_h + 0.9
    thumb_h = 2.75 if thumbs else 0.0
    height = 1.15 + 1.45 + 0.55 + chart_h + (0.7 + thumb_h if thumbs else 0.2) + 0.45
    fig = plt.figure(figsize=(width, height), dpi=150, facecolor=INK["surface"])
    top = lambda y: 1 - y / height  # noqa: E731  从顶端量的英寸 → 画布纵坐标

    count = len(samples)
    fig.text(0.04, top(0.55), f"地图分类模型更新：{online.name} → {candidate.name}", fontsize=22, color=INK["primary"])
    fig.text(0.04, top(0.95), (
        f"固定真实验证 {count} 张（地图类 {sum(l != 'None' for l in labels)}、无小地图 {sum(l == 'None' for l in labels)}）"
        f" · 线上 {online.md5} · 候选 {candidate.md5}" + (f" · 基线 {baseline.name}" if baseline else "")
    ), fontsize=12, color=INK["secondary"])

    correct_old, correct_new = (sum(top1 == label for top1, label in zip(scores[m.name + ":top"], labels)) for m in (online, candidate))
    tiles = [
        ("认对", f"{correct_new}/{count}", f"线上 {correct_old}/{count}", delta_text(correct_new - correct_old, 0)),
        ("正确类平均置信度", f"{new.mean():.2f}%", f"线上 {old.mean():.2f}%", delta_text(new.mean() - old.mean(), 2)),
        ("最低置信度", f"{new.min():.1f}%", f"线上 {old.min():.1f}%", delta_text(new.min() - old.min())),
    ]
    if baseline:
        below = lambda s: int(np.sum(s < base - GATE_MARGIN_PT))  # noqa: E731
        tiles.append((f"低于 {baseline.name} 超 {GATE_MARGIN_PT:g} 个百分点", f"{below(new)} 张", f"线上 {below(old)} 张",
                      delta_text(below(new) - below(old), 0, lower_is_better=True)))
    tile_w = 0.92 / len(tiles)
    for k, tile in enumerate(tiles):
        draw_tile(fig, [0.04 + k * tile_w, top(2.5), tile_w - 0.01, 1.3 / height], *tile)

    y0 = 2.5 + 0.55
    fig.text(0.04, top(y0 - 0.1), f"正确类置信度有变化的样本（{len(rows)}/{count}）", fontsize=15, color=INK["primary"])
    ax = fig.add_axes([0.36, top(y0 + chart_h - 0.35), 0.50, (chart_h - 0.75) / height])
    ys = np.arange(len(rows))[::-1]
    for y, i in zip(ys, rows):
        ax.plot([old[i], new[i]], [y, y], color=INK["axis"], linewidth=2, zorder=1, solid_capstyle="round")
        if base is not None and not np.isnan(base[i]):
            ax.plot(base[i], y, marker="|", markersize=13, markeredgewidth=2, color=INK["secondary"], zorder=2)
        ax.plot(old[i], y, "o", markersize=8, color=ONLINE_COLOR, markeredgecolor=INK["surface"], markeredgewidth=1.5, zorder=3)
        ax.plot(new[i], y, "o", markersize=9, color=CANDIDATE_COLOR, markeredgecolor=INK["surface"], markeredgewidth=1.5, zorder=4)
        text, color = delta_text(new[i] - old[i])
        ax.text(1.02, y, text, transform=ax.get_yaxis_transform(), fontsize=11, color=color, va="center")
    ax.set_yticks(ys, [f"{regions[i]} · {source_label(samples[i].path)}" for i in rows], fontsize=11, color=INK["primary"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], fontsize=10, color=INK["muted"])
    ax.grid(axis="x", color=INK["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK["axis"])
    ax.set_facecolor(INK["surface"])
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=ONLINE_COLOR, label=f"线上 {online.name}"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=CANDIDATE_COLOR, label=f"候选 {candidate.name}"),
    ]
    if baseline:
        handles.append(plt.Line2D([], [], marker="|", linestyle="", markersize=13, markeredgewidth=2,
                                  color=INK["secondary"], label=f"基线 {baseline.name}"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), frameon=False,
              fontsize=11, labelcolor=INK["secondary"], handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
    rest = count - len(rows)
    fig.text(0.36, top(y0 + chart_h + 0.12),
             f"其余 {rest} 张两版都在 {LOW_SCORE_PT:g}% 以上且变化不足 {CHANGE_MIN_PT:g} 个百分点，未列出。",
             fontsize=10, color=INK["muted"])

    if thumbs:
        y1 = y0 + chart_h + 0.7
        fig.text(0.04, top(y1 - 0.1), "变化最大的样本", fontsize=15, color=INK["primary"])
        cell = 0.92 / THUMBNAIL_COUNT
        for k, i in enumerate(thumbs):
            left = 0.04 + k * cell
            img_ax = fig.add_axes([left + cell * 0.12, top(y1 + 1.55), cell * 0.76, 1.35 / height])
            img_ax.imshow(cv2.cvtColor(samples[i].image, cv2.COLOR_BGR2RGB), interpolation="nearest")
            img_ax.set_axis_off()
            cx = left + cell / 2
            fig.text(cx, top(y1 + 1.8), regions[i].split("（")[0], fontsize=11, color=INK["primary"], ha="center")
            # 缩略图下只放区域与来源，格号留在上面的逐样本对比里
            fig.text(cx, top(y1 + 2.05), source_label(samples[i].path), fontsize=10, color=INK["secondary"], ha="center")
            fig.text(cx, top(y1 + 2.35), f"{old[i]:.1f}% → {new[i]:.1f}%", fontsize=12, color=INK["primary"], ha="center")
            for line, (role, model) in enumerate((("线上", online), ("候选", candidate))):
                top1 = scores[model.name + ":top"][i]
                if top1 != labels[i]:
                    fig.text(cx, top(y1 + 2.6 + line * 0.22), f"{role}判成 {namer(top1).split('（')[0]}",
                             fontsize=9, color=INK["muted"], ha="center")

    fig.text(0.04, top(height - 0.25), (
        f"置信度为模型给正确类别的概率；固定验证帧取自实机截图，按推理管线预处理到 128×128。"
        f" 生成于 {datetime.date.today():%Y-%m-%d} · report.py"
    ), fontsize=9, color=INK["muted"])
    fig.savefig(out, facecolor=INK["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="地图分类模型更新报告（固定真实验证帧对比）")
    parser.add_argument("--online", required=True, help="线上模型：ONNX 路径或 MaaEnd-AI 提交号")
    parser.add_argument("--candidate", required=True, help="候选模型：ONNX 路径或 MaaEnd-AI 提交号")
    parser.add_argument("--online-name", default=None, help="报告里线上模型的名字（默认用路径/提交号）")
    parser.add_argument("--candidate-name", default=None, help="报告里候选模型的名字（默认用路径/提交号）")
    parser.add_argument("--baseline", default=DEFAULT_OPTIONS["baseline"],
                        help=f"逐样本门槛基线，传 none 关闭 (default: {DEFAULT_OPTIONS['baseline']}，即 Aug11)")
    parser.add_argument("--baseline-name", default="Aug11", help="基线在报告里的名字 (default: Aug11)")
    parser.add_argument("--model-repo", default=DEFAULT_OPTIONS["model_repo"], help="按提交号取模型的 MaaEnd-AI 仓库")
    parser.add_argument("--maaend", default=DEFAULT_OPTIONS["maaend"], help="MaaEnd 仓库，用于把格号换成区域名")
    parser.add_argument("--validation", default=DEFAULT_OPTIONS["validation"], help="固定验证帧目录")
    parser.add_argument("--out", default=DEFAULT_OPTIONS["out"], help="输出 PNG 路径")
    args = parser.parse_args()

    repo = Path(args.model_repo).expanduser()
    models = [
        load_model(args.online, args.online_name, repo),
        load_model(args.candidate, args.candidate_name, repo),
        None if args.baseline.lower() == "none" else load_model(args.baseline, args.baseline_name, repo),
    ]
    samples = load_samples(Path(args.validation))
    labels = [sample.label for sample in samples]
    scores = {}
    for model in filter(None, models):
        probs = [model.scores(sample.image) for sample in samples]
        scores[model.name] = np.array([
            p[model.classes.index(label)] * 100 if label in model.classes else np.nan
            for p, label in zip(probs, labels)
        ])
        scores[model.name + ":top"] = [model.classes[int(np.argmax(p))] for p in probs]

    namer = RegionNamer(Path(args.maaend).expanduser(), Path(DEFAULT_OPTIONS["source"]))
    use_cjk_font()
    render(samples, namer, models, scores, Path(args.out))
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
