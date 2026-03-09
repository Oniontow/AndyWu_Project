from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def find_hdf5_files(dataset_dir: Path) -> list[Path]:
	files = sorted(dataset_dir.rglob("*.h5")) + sorted(dataset_dir.rglob("*.hdf5"))
	return files


def resolve_target_files(dataset_dir: Path, targets: Iterable[str]) -> list[Path]:
	resolved_files: list[Path] = []
	for target in targets:
		candidate = dataset_dir / target
		if candidate.exists() and candidate.is_file():
			resolved_files.append(candidate)
		else:
			print(f"[Skip] 找不到指定檔案: {candidate}")
	return resolved_files


def _collect_numeric_datasets(group: h5py.Group, prefix: str = "") -> list[tuple[str, h5py.Dataset]]:
	found: list[tuple[str, h5py.Dataset]] = []
	for name, item in group.items():
		full_name = f"{prefix}/{name}" if prefix else name
		if isinstance(item, h5py.Dataset):
			if np.issubdtype(item.dtype, np.number):
				found.append((full_name, item))
		elif isinstance(item, h5py.Group):
			found.extend(_collect_numeric_datasets(item, full_name))
	return found


def _to_2d_array(data: np.ndarray) -> np.ndarray:
	if data.ndim == 1:
		return data.reshape(-1, 1)
	if data.ndim == 2:
		return data
	return data.reshape(data.shape[0], -1)


def select_feature_dataset(h5_file: h5py.File) -> tuple[str, np.ndarray]:
	numeric_datasets = _collect_numeric_datasets(h5_file)
	if not numeric_datasets:
		raise ValueError("找不到任何數值型 dataset。")

	best_name = ""
	best_data: np.ndarray | None = None
	best_score = -1

	for name, ds in numeric_datasets:
		try:
			arr = ds[()]
		except Exception:
			continue
		if arr.size == 0:
			continue

		arr = np.asarray(arr)
		arr_2d = _to_2d_array(arr)
		if arr_2d.shape[0] < 2:
			continue

		score = arr_2d.shape[0] * arr_2d.shape[1]
		if score > best_score:
			best_score = score
			best_name = name
			best_data = arr_2d

	if best_data is None:
		raise ValueError("數值型 dataset 存在，但沒有可用特徵。")

	return best_name, best_data


def load_features_from_file(path: Path) -> tuple[np.ndarray, str]:
	with h5py.File(path, "r") as f:
		ds_name, features = select_feature_dataset(f)
	print(f"[OK] {path.name} -> '{ds_name}', shape={features.shape}")
	return features, ds_name


def run_tsne(
	x: np.ndarray,
	random_state: int,
	perplexity: float,
	n_iter: int,
) -> np.ndarray:
	x = np.nan_to_num(x, copy=False)
	x = StandardScaler().fit_transform(x)

	tsne = TSNE(
		n_components=2,
		perplexity=perplexity,
		max_iter=n_iter,
		init="pca",
		learning_rate="auto",
		random_state=random_state,
		verbose=1,
	)
	return tsne.fit_transform(x)


def plot_embedding(
	emb: np.ndarray,
	output_path: Path,
	title: str,
	show_plot: bool,
) -> None:
	plt.figure(figsize=(10, 8))
	plt.scatter(emb[:, 0], emb[:, 1], s=7, alpha=0.7)
	plt.title(title)
	plt.xlabel("t-SNE 1")
	plt.ylabel("t-SNE 2")
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	print(f"圖已儲存：{output_path}")
	if show_plot:
		plt.show()
	else:
		plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="把 Last.fm HDF5 資料做 t-SNE 2D 視覺化")
	default_data_dir = Path(__file__).resolve().parent
	parser.add_argument(
		"--data_dir",
		type=Path,
		default=default_data_dir,
		help=f"HDF5 資料夾路徑（預設: {default_data_dir}）",
	)
	parser.add_argument("--output", type=Path, default=Path("./lastfm_tsne.png"), help="輸出圖片檔名")
	parser.add_argument("--max_samples", type=int, default=5000, help="最多抽樣點數（0 表示不抽樣）")
	parser.add_argument(
		"--targets",
		nargs="+",
		default=None,
		help="指定相對於 data_dir 的 HDF5 檔案路徑（未指定時預設跑全部）",
	)
	parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
	parser.add_argument("--n_iter", type=int, default=1000, help="t-SNE 訓練迭代次數")
	parser.add_argument("--seed", type=int, default=42, help="隨機種子")
	parser.add_argument("--show", action="store_true", help="是否顯示視窗")
	return parser


def build_output_path(base_output: Path, dataset_file: Path, total_files: int) -> Path:
	suffix = base_output.suffix if base_output.suffix else ".png"
	if total_files == 1:
		return base_output.with_suffix(suffix)
	return base_output.with_name(f"{base_output.stem}_{dataset_file.stem}{suffix}")


def main() -> None:
	args = build_arg_parser().parse_args()

	if not args.data_dir.exists():
		raise FileNotFoundError(f"找不到資料夾：{args.data_dir}")

	if args.targets:
		hdf5_files = resolve_target_files(args.data_dir, args.targets)
		if not hdf5_files:
			print("指定目標檔案都不存在，改為掃描整個 data_dir 下的 .h5/.hdf5")
			hdf5_files = find_hdf5_files(args.data_dir)
	else:
		print("未指定 --targets，預設掃描並使用全部 .h5/.hdf5")
		hdf5_files = find_hdf5_files(args.data_dir)
	if not hdf5_files:
		raise FileNotFoundError(f"在 {args.data_dir} 找不到 .h5 或 .hdf5 檔案")

	print(f"找到 {len(hdf5_files)} 個 HDF5 檔，將逐一作圖...")

	for file_path in hdf5_files:
		try:
			x, ds_name = load_features_from_file(file_path)
		except Exception as err:
			print(f"[Skip] {file_path.name}: {err}")
			continue

		print(f"處理 {file_path.name}: 樣本數={x.shape[0]}, 維度={x.shape[1]}")

		if args.max_samples > 0 and x.shape[0] > args.max_samples:
			rng = np.random.default_rng(args.seed)
			indices = rng.choice(x.shape[0], size=args.max_samples, replace=False)
			x = x[indices]
			print(f"{file_path.name} 已抽樣到 {x.shape[0]} 個點")

		perplexity = min(args.perplexity, max(5.0, (x.shape[0] - 1) / 3.0))
		print(f"{file_path.name} 使用 perplexity={perplexity:.2f}")

		emb = run_tsne(x, random_state=args.seed, perplexity=perplexity, n_iter=args.n_iter)
		output_path = build_output_path(args.output, file_path, total_files=len(hdf5_files))
		plot_embedding(
			emb,
			output_path=output_path,
			title=f"{file_path.stem}:{ds_name} t-SNE (2D)",
			show_plot=args.show,
		)


if __name__ == "__main__":
	main()
