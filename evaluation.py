# import json
# import glob
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def load_json(file_path: str) -> dict:
#     """JSONファイルを読み込んで辞書を返す"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return data

# def main(gt_path, algo_dirs):
#     # ==========================
#     # 1. GT(真値) を読み込み
#     # ==========================
#     gt_data = load_json(gt_path)
#     # 例: gt_data = {"group1": 10.0, "group2": 150.0, "group3": 0.85}

#     # ==========================
#     # 2. algo1, algo2 の結果をまとめて読み込み
#     # ==========================
#     # algo1/result_XX.json のパスを探して読み込む
    

#     algo_paths = {}
#     algo_values = {}
#     for algo_dir in algo_dirs:
#         algo_dir = Path(algo_dir)
#         algo_files = sorted(glob.glob(os.path.join(algo_dir, "result_*.json")))
#         algo_paths[algo_dir.stem] = algo_files
#         algo_values[algo_dir.stem]

#     #algo1_files = sorted(glob.glob("algo1/result_*.json"))
#     #algo2_files = sorted(glob.glob("algo2/result_*.json"))

#     # それぞれのグループについて、algo1, algo2の値を保持するリストを用意
#     group_names = ["group1", "group2", "group3"]
#     # ユニットは仮で記載しています。実際の単位に置き換えてください。
#     group_units = ["unit(G1)", "unit(G2)", "unit(G3)"]

#     algo1_values = {g: [] for g in group_names}
#     algo2_values = {g: [] for g in group_names}

#     # algo1の各ファイルを読み込み、group1, group2, group3の推定結果を追加
#     for f in algo1_files:
#         data = load_json(f)
#         for g in group_names:
#             algo1_values[g].append(data[g])

#     # algo2の各ファイルを読み込み、group1, group2, group3の推定結果を追加
#     for f in algo2_files:
#         data = load_json(f)
#         for g in group_names:
#             algo2_values[g].append(data[g])

#     # ==========================
#     # 3. グラフ作成の下準備
#     # ==========================
#     # 各グループごとに、GT, algo1(mean±std), algo2(mean±std) を描画する
#     # 棒グラフは各グループでサブプロットを作り、縦軸を変える想定

#     fig, axes = plt.subplots(nrows=1, ncols=len(group_names), figsize=(15, 5))
#     # サブプロットが1列の場合、axes は配列になる
#     if len(group_names) == 1:
#         axes = [axes]  # サブプロットが1つしかない場合でもリスト化

#     for i, g in enumerate(group_names):
#         ax = axes[i]

#         # GT値 (真値) は単一の値とし、エラーバーはなしと想定
#         gt_val = gt_data[g]

#         # algo1
#         a1_vals = np.array(algo1_values[g])
#         a1_mean = a1_vals.mean() if len(a1_vals) > 0 else np.nan
#         a1_std  = a1_vals.std()  if len(a1_vals) > 0 else np.nan

#         # algo2
#         a2_vals = np.array(algo2_values[g])
#         a2_mean = a2_vals.mean() if len(a2_vals) > 0 else np.nan
#         a2_std  = a2_vals.std()  if len(a2_vals) > 0 else np.nan

#         # 棒グラフに表示するデータとエラーバーを用意
#         bar_labels = ["GT", "algo1", "algo2"]
#         bar_means = [gt_val, a1_mean, a2_mean]
#         bar_stds  = [0.0, a1_std, a2_std]  # GTにはエラーバー0としている

#         # x座標を確保
#         x = np.arange(len(bar_labels))

#         # 棒グラフの描画
#         bars = ax.bar(x, bar_means, yerr=bar_stds, capsize=5, alpha=0.7)

#         # 軸ラベルやタイトル
#         ax.set_xticks(x)
#         ax.set_xticklabels(bar_labels)
#         ax.set_ylabel(f"Value [{group_units[i]}]")
#         ax.set_title(f"Comparison for {g}")

#         # 数値ラベルを棒グラフの上に描画 (オプション)
#         for rect, mean_val in zip(bars, bar_means):
#             height = rect.get_height()
#             ax.text(
#                 rect.get_x() + rect.get_width()/2.0,
#                 height,
#                 f"{mean_val:.2f}",
#                 ha="center", va="bottom"
#             )

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()



import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import matplotlib.cm as cm

def load_json(file_path: str) -> dict:
    """
    JSONファイルを読み込み、辞書として返す
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(gt_path: str, algo_dirs: list):
    """
    GT(JSONファイル)と、任意個数のアルゴリズムディレクトリを受け取り、
    3つ以上のグループに分かれたデータの真値と推定結果(平均・標準偏差)を
    サブプロットでエラーバー付きの棒グラフとして描画する。

    Parameters
    ----------
    gt_path : str
        真値(GT)が格納されたJSONファイルへのパス
    algo_dirs : list
        アルゴリズムごとの結果ディレクトリのリスト
        例: ["algo1", "algo2", "algo3", ...]
        各ディレクトリには result_XX.json が複数存在し、
        グループ名→推定値を持つJSONファイルであることを想定
    """
    # ==========================
    # 1. GT(真値) を読み込み & グループ取得
    # ==========================
    gt_data_ = load_json(gt_path)
    # 例: gt_data = {"group1": 10.0, "group2": 150.0, "group3": 0.85}
    #group_names = list(gt_data.keys())
    group_names = ["reproj_error", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]
    gt_data = {}
    gt_data["reproj_error"] = 0.0
    gt_data["fx"] = gt_data_["intrin"]["fx"]
    gt_data["fy"] = gt_data_["intrin"]["fy"]
    gt_data["cx"] = gt_data_["intrin"]["cx"]
    gt_data["cy"] = gt_data_["intrin"]["cy"]
    gt_data["k1"] = gt_data_["intrin"]["k1"]
    gt_data["k2"] = gt_data_["intrin"]["k2"]
    gt_data["k3"] = gt_data_["intrin"]["k3"]
    gt_data["p1"] = gt_data_["intrin"]["p1"]
    gt_data["p2"] = gt_data_["intrin"]["p2"]
    

    # 必要に応じてソートする場合
    # group_names.sort()

    # (オプション) グループの単位を管理したい場合は下記を使うなど
    # group_units = {
    #     "group1": "unit(G1)",
    #     "group2": "unit(G2)",
    #     "group3": "unit(G3)"
    # }
    # 今回は例としてNoneで扱います
    group_units = {g: "" for g in group_names}

    # ==========================
    # 2. アルゴリズム結果の読み込み
    # ==========================
    # アルゴリズム名 -> { グループ名 -> [値, 値, ...] } の形で格納する
    algo_results = {}

    for algo_dir in algo_dirs:
        algo_name = os.path.basename(algo_dir)  # ディレクトリ名をアルゴリズム名とする

        for setting in ["_img", "_gt"]:
            algo_results[algo_name + setting] = {g: [] for g in group_names}

            result_files = []
            for algo_trial in sorted(Path(algo_dir).iterdir()):
                if algo_trial.is_dir():
                    if setting == "_gt":
                        calib_json_path = algo_trial / "calibration.json"
                    else:
                        calib_json_path = algo_trial / "detected" / "calibration.json"
                    result_files.append(calib_json_path)

            for rf in result_files:
                data = load_json(rf)
                reproj_error = float(data["ret"])
                fx = float(data["mtx"][0][0])
                fy = float(data["mtx"][1][1])
                cx = float(data["mtx"][0][2])
                cy = float(data["mtx"][1][2])
                k1 = float(data["dist"][0][0])
                k2 = float(data["dist"][0][1])
                k3 = float(data["dist"][0][4])
                p1 = float(data["dist"][0][2])
                p2 = float(data["dist"][0][3])
                
                algo_results[algo_name + setting]["reproj_error"].append(reproj_error)
                algo_results[algo_name + setting]["fx"].append(fx)
                algo_results[algo_name + setting]["fy"].append(fy)
                algo_results[algo_name + setting]["cx"].append(cx)
                algo_results[algo_name + setting]["cy"].append(cy)
                algo_results[algo_name + setting]["k1"].append(k1)
                algo_results[algo_name + setting]["k2"].append(k2)
                algo_results[algo_name + setting]["k3"].append(k3)
                algo_results[algo_name + setting]["p1"].append(p1)
                algo_results[algo_name + setting]["p2"].append(p2)
                # # 取得したファイルの中に該当グループがあれば格納
                # for g in group_names:
                #     if g in data:
                #         algo_results[algo_name + setting][g].append(data[g])

    # ==========================
    # 3. グラフ作成の下準備
    # ==========================
    # subplotを "グループの数" だけ用意
    fig, axes = plt.subplots(nrows=1, ncols=len(group_names), figsize=(5 * len(group_names), 5))
    if len(group_names) == 1:
        # groupが1つしかない場合、axesをリストに
        axes = [axes]

    # 描画用のアルゴリズム名リスト (順番を固定したい場合はここでソート可)
    algo_names_sorted = sorted(algo_results.keys())

    # ==========================
    # 4. 各グループごとに GT と各アルゴの平均±標準偏差を可視化
    # ==========================
    for i, g in enumerate(group_names):
        ax = axes[i]

        # GT
        gt_val = gt_data[g]

        # 各アルゴの平均・標準偏差を算出
        means = []
        stds = []
        labels = []

        # # GTを先頭に追加
        # gt_val = gt_data[g]
        # labels.append("GT")
        # means.append(gt_val)
        # stds.append(0.0)  # GTにはエラーバーなし

        # アルゴリズムごとに平均・標準偏差を追加
        for algo_name in algo_names_sorted:
            vals = np.array(algo_results[algo_name][g]) - gt_val
            if len(vals) > 0:
                m = vals.mean()
                s = vals.std()
            else:
                m = np.nan
                s = 0.0
            labels.append(algo_name)
            means.append(m)
            stds.append(s)
            print(algo_name, s)

        x = np.arange(len(labels))  # x軸の位置
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(f"Value [{group_units[g]}]")
        ax.set_title(f"Group: {g}")

        # バーの上に数値を描画する
        for rect, mean_val in zip(bars, means):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width()/2.0,
                height,
                f"{mean_val:.2f}",
                ha="center", va="bottom"
            )

    plt.tight_layout()
    plt.show()



def main(gt_path: str, algo_dirs: list):
    """
    GT(JSONファイル)と任意個数のアルゴリズムディレクトリを受け取り、
    カメラパラメータや再投影誤差 (複数グループ) に対して、
    真値との差分の平均・標準偏差をエラーバー付き棒グラフで描画する。

    Parameters
    ----------
    gt_path : str
        真値(GT)が格納されたJSONファイルへのパス。
        JSON 例:
        {
          "intrin": {
            "fx": ...,
            "fy": ...,
            "cx": ...,
            "cy": ...,
            "k1": ...,
            "k2": ...,
            "k3": ...,
            "p1": ...,
            "p2": ...
          }
        }
    algo_dirs : list
        アルゴリズムごとの結果ディレクトリのリスト
        例: ["algo1", "algo2", "algo3", ...]
        各ディレクトリは複数のサブフォルダを持ち、以下のような構造を想定:
            algo1/
              ├─ trial_01/
              │     ├─ calibration.json (ret, mtx, dist...)
              │     └─ detected/
              │           └─ calibration.json (...)
              ├─ trial_02/
              │     ├─ calibration.json
              │     └─ detected/
              │           └─ calibration.json
              ...
    """

    # ==========================
    # 1. GT(真値) を読み込み & グループ取得
    # ==========================
    gt_data_ = load_json(gt_path)
    # 真値のキーをリスト化
    group_names = [
        "reproj_error", "fx", "fy", "cx", "cy",
        "k1", "k2", "k3", "p1", "p2"
    ]

    # 真値を整理して新しい辞書 gt_data に格納
    gt_data = {}
    gt_data["reproj_error"] = 0.0
    gt_data["fx"] = gt_data_["intrin"]["fx"]
    gt_data["fy"] = gt_data_["intrin"]["fy"]
    gt_data["cx"] = gt_data_["intrin"]["cx"]
    gt_data["cy"] = gt_data_["intrin"]["cy"]
    gt_data["k1"] = gt_data_["intrin"]["k1"]
    gt_data["k2"] = gt_data_["intrin"]["k2"]
    gt_data["k3"] = gt_data_["intrin"]["k3"]
    gt_data["p1"] = gt_data_["intrin"]["p1"]
    gt_data["p2"] = gt_data_["intrin"]["p2"]

    # (オプション) グループの単位を管理したい場合はここで設定可
    group_units = {g: "" for g in group_names}

    # ==========================
    # 2. アルゴリズム結果の読み込み
    # ==========================
    # アルゴリズム名 -> { グループ名 -> [値, 値, ...] } の形で格納する
    algo_results = {}

    # 各アルゴリズムディレクトリについて処理
    for algo_dir in algo_dirs:
        # ディレクトリ名をアルゴリズム名とする
        algo_name = os.path.basename(algo_dir)

        # 2種類のcalibration.jsonパスを抽出するため "_img" と "_gt" などのタグをつける
        for setting in ["_img", "_gt"]:
            # settingごとに辞書作成
            key = algo_name + setting
            algo_results[key] = {g: [] for g in group_names}

            # すべての試行フォルダを探す
            result_files = []
            for algo_trial in sorted(Path(algo_dir).iterdir()):
                if algo_trial.is_dir():
                    if setting == "_gt":
                        # 例: trial_01/calibration.json
                        calib_json_path = algo_trial / "calibration.json"
                    else:
                        # 例: trial_01/detected/calibration.json
                        calib_json_path = algo_trial / "detected" / "calibration.json"
                    result_files.append(calib_json_path)

            # calibration.json を読み込む
            for rf in result_files:
                data = load_json(rf)

                # JSON構造から必要な値を取り出し
                reproj_error = float(data["ret"])
                fx = float(data["mtx"][0][0])
                fy = float(data["mtx"][1][1])
                cx = float(data["mtx"][0][2])
                cy = float(data["mtx"][1][2])
                k1 = float(data["dist"][0][0])
                k2 = float(data["dist"][0][1])
                k3 = float(data["dist"][0][4])
                p1 = float(data["dist"][0][2])
                p2 = float(data["dist"][0][3])

                # グループごとに値を追加
                algo_results[key]["reproj_error"].append(reproj_error)
                algo_results[key]["fx"].append(fx)
                algo_results[key]["fy"].append(fy)
                algo_results[key]["cx"].append(cx)
                algo_results[key]["cy"].append(cy)
                algo_results[key]["k1"].append(k1)
                algo_results[key]["k2"].append(k2)
                algo_results[key]["k3"].append(k3)
                algo_results[key]["p1"].append(p1)
                algo_results[key]["p2"].append(p2)

    # ==========================
    # 3. グラフ作成の下準備
    # ==========================
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(group_names),
        figsize=(5 * len(group_names), 5)
    )
    if len(group_names) == 1:
        axes = [axes]

    # アルゴリズム名のソート
    algo_names_sorted = sorted(algo_results.keys())

    # カラーマップ (tab10 を使用し、アルゴ数だけ色を取る)
    cmap = cm.get_cmap("tab10", len(algo_names_sorted))
    colors = [cmap(i) for i in range(len(algo_names_sorted))]

    # 図全体のタイトル
    fig.suptitle("Differences from GT", fontsize=16, y=1.02)




    # ==========================
    # 4. グラフ描画: 真値との差分 (平均±標準偏差)
    # ==========================
    for i, g in enumerate(group_names):
        ax = axes[i]
        gt_val = gt_data[g]

        means = []
        stds = []
        labels = []

        for algo_name in algo_names_sorted:
            vals = np.array(algo_results[algo_name][g]) - gt_val
            if len(vals) > 0:
                m = vals.mean()
                s = vals.std()
            else:
                m = np.nan
                s = 0.0
            means.append(m)
            stds.append(s)
            labels.append(algo_name)

        x = np.arange(len(labels))

        # バーを1つずつ描画
        for idx in range(len(labels)):
            bar = ax.bar(
                x[idx],
                means[idx],
                yerr=stds[idx],
                capsize=4,
                alpha=0.8,
                color=colors[idx],
                label=labels[idx]  # 凡例用ラベル
            )

        # 真値との差なので、0 を基準とする水平線を追加
        ax.axhline(
            y=0,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7
        )

        # 軸ラベル・タイトル
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(f"Δ Value [{group_units[g]}]")
        ax.set_title(f"{g}", fontsize=12)

        # グリッドを追加 (y軸方向)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # バーの上に数値を表示 (mean)
        for rect, mean_val in zip(ax.patches, means):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        # 凡例の表示 (subplotごとに同じなら1回でいいが、簡易実装で各軸につける)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()




def main(gt_path: str, algo_dirs: list):
    """
    GT(JSONファイル)と任意個数のアルゴリズムディレクトリを受け取り、
    カメラパラメータや再投影誤差に対して、
    真値との差分の平均・標準偏差をエラーバー付き棒グラフで描画する。

    今回は下記3つのグループにまとめて表示する:
      1) ["reproj_error"] (単独)
      2) ["fx", "fy", "cx", "cy"]
      3) ["k1", "k2", "k3", "p1", "p2"]
    """

    # ==========================
    # 1. GT(真値) を読み込み & グループの定義
    # ==========================
    gt_data_ = load_json(gt_path)

    # グラフ表示をまとめるために3つのグループを定義
    param_groups = [
        ["reproj_error"],                # 1つだけ
        ["fx", "fy", "cx", "cy"],        # 4つを1グラフに
        ["k1", "k2", "k3", "p1", "p2"]   # 5つを1グラフに
    ]

    # 真値データ (gt_data) を展開
    gt_data = {}
    gt_data["reproj_error"] = 0.0
    gt_data["fx"] = gt_data_["intrin"]["fx"]
    gt_data["fy"] = gt_data_["intrin"]["fy"]
    gt_data["cx"] = gt_data_["intrin"]["cx"]
    gt_data["cy"] = gt_data_["intrin"]["cy"]
    gt_data["k1"] = gt_data_["intrin"]["k1"]
    gt_data["k2"] = gt_data_["intrin"]["k2"]
    gt_data["k3"] = gt_data_["intrin"]["k3"]
    gt_data["p1"] = gt_data_["intrin"]["p1"]
    gt_data["p2"] = gt_data_["intrin"]["p2"]

    # (任意) 単位を設定したい場合はここで管理
    group_units = {
        "reproj_error": "",
        "fx": "",
        "fy": "",
        "cx": "",
        "cy": "",
        "k1": "",
        "k2": "",
        "k3": "",
        "p1": "",
        "p2": ""
    }

    # ==========================
    # 2. アルゴリズム結果の読み込み
    # ==========================
    # { "algo名_設定": { "パラメータ名": [値, 値, ...], ... }, ... }
    algo_results = {}

    for algo_dir in algo_dirs:
        algo_name = os.path.basename(algo_dir)

        # "_img", "_gt" など 2種類のcalibration.jsonを比較したい想定
        # 必要なければ1種類にまとめてもOK
        for setting in ["_img", "_gt"]:
            key = algo_name + setting
            algo_results[key] = {
                "reproj_error": [],
                "fx": [], "fy": [], "cx": [], "cy": [],
                "k1": [], "k2": [], "k3": [], "p1": [], "p2": []
            }

            # trial_* フォルダを走査して calibration.json を探す
            for algo_trial in sorted(Path(algo_dir).iterdir()):
                if not algo_trial.is_dir():
                    continue

                if setting == "_gt":
                    # 例: trial_01/calibration.json
                    calib_json_path = algo_trial / "calibration.json"
                else:
                    # 例: trial_01/detected/calibration.json
                    calib_json_path = algo_trial / "detected" / "calibration.json"

                # 存在する場合のみ読み込む
                if calib_json_path.exists():
                    data = load_json(calib_json_path)

                    # 必要な値を抽出
                    reproj_error = float(data["ret"])
                    fx = float(data["mtx"][0][0])
                    fy = float(data["mtx"][1][1])
                    cx = float(data["mtx"][0][2])
                    cy = float(data["mtx"][1][2])
                    k1 = float(data["dist"][0][0])
                    k2 = float(data["dist"][0][1])
                    k3 = float(data["dist"][0][4])
                    p1 = float(data["dist"][0][2])
                    p2 = float(data["dist"][0][3])

                    # リストに追加
                    algo_results[key]["reproj_error"].append(reproj_error)
                    algo_results[key]["fx"].append(fx)
                    algo_results[key]["fy"].append(fy)
                    algo_results[key]["cx"].append(cx)
                    algo_results[key]["cy"].append(cy)
                    algo_results[key]["k1"].append(k1)
                    algo_results[key]["k2"].append(k2)
                    algo_results[key]["k3"].append(k3)
                    algo_results[key]["p1"].append(p1)
                    algo_results[key]["p2"].append(p2)

    # ==========================
    # 3. グラフ作成の下準備
    # ==========================
    # 今回は param_groups の要素数(=3) だけサブプロットを作る
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(param_groups),
        figsize=(6 * len(param_groups), 5),
        squeeze=False  # 常に2次元配列で返す
    )
    axes = axes[0]  # 1行だけなので 0番目を取り出す → [axis0, axis1, axis2] になる

    # アルゴリズム名キーをソート
    algo_names_sorted = sorted(algo_results.keys())

    # カラーマップ (tab10) でアルゴ数だけ色を取得
    cmap = cm.get_cmap("tab10", len(algo_names_sorted))
    colors = [cmap(i) for i in range(len(algo_names_sorted))]

    # 図全体タイトル (任意)
    fig.suptitle("Differences from GT (Grouped Parameters)", fontsize=16, y=1.02)

    # ==========================
    # 4. グラフ描画 (グループごと)
    # ==========================
    # param_groupsに基づいて、各グループを1つのサブプロットにまとめる
    for subplot_index, param_list in enumerate(param_groups):
        ax = axes[subplot_index]

        # x 軸の位置: パラメータ数に応じて [0, 1, 2, ...]
        x_positions = np.arange(len(param_list))

        # 幅: パラメータごとに「アルゴ数」のバーを並べたいので
        # 例: param_listが4, algoが3個なら 4個のグループ × 3本 = 12本
        # 1つのグループ内でバーを少しずつ横にずらす計算
        total_width = 0.8
        width_per_algo = total_width / len(algo_names_sorted)


        # param_list内の各パラメータについて真値との差分の平均・標準偏差を算出
        for algo_i, algo_name in enumerate(algo_names_sorted):
            means = []
            stds = []

            for param in param_list:
                gt_val = gt_data[param]
                vals = np.array(algo_results[algo_name][param]) - gt_val

                if len(vals) > 0:
                    m = vals.mean()
                    s = vals.std()
                else:
                    m = np.nan
                    s = 0.0

                means.append(m)
                stds.append(s)

            # バーをずらすオフセットを計算
            offset = (algo_i - (len(algo_names_sorted) - 1) / 2) * width_per_algo

            # 実際の x 軸位置 (paramごとに少しずらし)
            x_for_this_algo = x_positions + offset

            # バーを描画
            ax.bar(
                x_for_this_algo,
                means,
                yerr=stds,
                width=width_per_algo * 0.9,  # バー幅
                color=colors[algo_i],
                alpha=0.8,
                capsize=4,
                label=algo_name
            )

            # バーの上に数値を描画
            for x_pos, mean_val in zip(x_for_this_algo, means):
                ax.text(
                    x_pos,
                    mean_val,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        # 0ライン(真値との差分なので分かりやすく)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

        # 軸ラベルの設定
        ax.set_xticks(x_positions)
        ax.set_xticklabels(param_list, rotation=45, ha="right")

        # サブプロットタイトル: 今回はパラメータリストを単純に表示
        # (カスタマイズしたい場合は "Intrinsic Params (fx, fy, cx, cy)" など適宜変更)
        ax.set_title(f"{', '.join(param_list)}", fontsize=12)

        # y 軸ラベル
        ax.set_ylabel("Δ Value")

        # グリッドを追加 (読みやすくするため)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # 凡例を追加 (最初のサブプロットのみ、あるいは全サブプロットに付けるなどお好みで)
        if subplot_index == 0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()



def main(gt_path: str, algo_dirs: list):
    """
    GT(JSONファイル)と任意個数のアルゴリズムディレクトリを受け取り、
    カメラパラメータや再投影誤差に対して、元の推定値(平均±標準偏差)を
    エラーバー付き棒グラフで描画し、各パラメータのGT(真値)を
    横線(hline)として表示する。

    今回は下記3つのグループにまとめて表示する:
      1) ["reproj_error"] (単独)
      2) ["fx", "fy", "cx", "cy"]
      3) ["k1", "k2", "k3", "p1", "p2"]
    """

    # ==========================
    # 1. GT(真値) を読み込み & グループの定義
    # ==========================
    gt_data_ = load_json(gt_path)

    # グラフでまとめるパラメータ群
    param_groups = [
        ["reproj_error"],                # グループ1: 再投影誤差(1つだけ)
        ["fx", "fy", "cx", "cy"],        # グループ2: fx, fy, cx, cy
        ["k1", "k2", "k3", "p1", "p2"]   # グループ3: 歪み係数たち
    ]

    # 真値を取り出し (キーを対応付け)
    gt_data = {}
    gt_data["reproj_error"] = 0.0
    gt_data["fx"] = gt_data_["intrin"]["fx"]
    gt_data["fy"] = gt_data_["intrin"]["fy"]
    gt_data["cx"] = gt_data_["intrin"]["cx"]
    gt_data["cy"] = gt_data_["intrin"]["cy"]
    gt_data["k1"] = gt_data_["intrin"]["k1"]
    gt_data["k2"] = gt_data_["intrin"]["k2"]
    gt_data["k3"] = gt_data_["intrin"]["k3"]
    gt_data["p1"] = gt_data_["intrin"]["p1"]
    gt_data["p2"] = gt_data_["intrin"]["p2"]

    # (オプション) 単位を入れたいならここで param → unit のマッピングを定義
    group_units = {
        "reproj_error": "",
        "fx": "",
        "fy": "",
        "cx": "",
        "cy": "",
        "k1": "",
        "k2": "",
        "k3": "",
        "p1": "",
        "p2": ""
    }

    # ==========================
    # 2. アルゴリズム結果の読み込み
    # ==========================
    # { "algo名_設定": { param: [値, 値, ...], ... }, ... }
    algo_results = {}

    for algo_dir in algo_dirs:
        algo_name = os.path.basename(algo_dir)

        # "_img", "_gt" など複数の結果パターンを使うなら以下
        # （不要なら1つにまとめてもOKです）
        for setting in ["_img", "_gt"]:
            key = algo_name + setting
            algo_results[key] = {
                "reproj_error": [],
                "fx": [], "fy": [], "cx": [], "cy": [],
                "k1": [], "k2": [], "k3": [], "p1": [], "p2": []
            }

            # algo_dir配下のサブフォルダ(例: trial_01, trial_02...)を走査
            for algo_trial in sorted(Path(algo_dir).iterdir()):
                if not algo_trial.is_dir():
                    continue

                if setting == "_gt":
                    calib_json_path = algo_trial / "calibration.json"
                else:
                    calib_json_path = algo_trial / "detected" / "calibration.json"

                if calib_json_path.exists():
                    data = load_json(calib_json_path)

                    # JSON構造から必要な値を抽出
                    reproj_error = float(data["ret"])
                    fx = float(data["mtx"][0][0])
                    fy = float(data["mtx"][1][1])
                    cx = float(data["mtx"][0][2])
                    cy = float(data["mtx"][1][2])
                    k1 = float(data["dist"][0][0])
                    k2 = float(data["dist"][0][1])
                    k3 = float(data["dist"][0][4])
                    p1 = float(data["dist"][0][2])
                    p2 = float(data["dist"][0][3])

                    # 辞書に格納
                    algo_results[key]["reproj_error"].append(reproj_error)
                    algo_results[key]["fx"].append(fx)
                    algo_results[key]["fy"].append(fy)
                    algo_results[key]["cx"].append(cx)
                    algo_results[key]["cy"].append(cy)
                    algo_results[key]["k1"].append(k1)
                    algo_results[key]["k2"].append(k2)
                    algo_results[key]["k3"].append(k3)
                    algo_results[key]["p1"].append(p1)
                    algo_results[key]["p2"].append(p2)

    # ==========================
    # 3. グラフ作成の下準備
    # ==========================
    # param_groups の要素数分だけサブプロットを作成
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(param_groups),
        figsize=(6 * len(param_groups), 5),
        squeeze=False
    )
    # axesを1行だけにする
    axes = axes[0]

    # アルゴリズム名キーをソート
    algo_names_sorted = sorted(algo_results.keys())

    # カラーマップ (tab10) でアルゴ数だけ色を取得
    cmap = cm.get_cmap("tab10", len(algo_names_sorted))
    colors = [cmap(i) for i in range(len(algo_names_sorted))]

    # 図全体タイトル (任意)
    fig.suptitle("Camera Parameters (Original Values) with GT lines", fontsize=16, y=1.02)

    # ==========================
    # 4. グラフ描画 (グループごと) - 元の値を描画
    # ==========================
    for subplot_index, param_list in enumerate(param_groups):
        ax = axes[subplot_index]

        # x 軸上の位置 (パラメータの個数分)
        x_positions = np.arange(len(param_list))

        # 棒幅などの設定
        total_width = 0.8
        width_per_algo = total_width / len(algo_names_sorted)

        for algo_i, algo_name in enumerate(algo_names_sorted):
            means = []
            stds = []

            for param in param_list:
                vals = np.array(algo_results[algo_name][param])
                if len(vals) > 0:
                    m = vals.mean()
                    s = vals.std()
                else:
                    m = np.nan
                    s = 0.0

                means.append(m)
                stds.append(s)

            # バーをずらすためのオフセット
            offset = (algo_i - (len(algo_names_sorted) - 1) / 2) * width_per_algo
            x_for_this_algo = x_positions + offset

            # エラーバー付きの棒を描画
            ax.bar(
                x_for_this_algo,
                means,
                yerr=stds,
                width=width_per_algo * 0.9,
                color=colors[algo_i],
                alpha=0.8,
                capsize=4,
                label=algo_name
            )

            # バーの上に数値を表示
            for x_pos, mean_val in zip(x_for_this_algo, means):
                ax.text(
                    x_pos,
                    mean_val,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        # ここからがポイント:
        # 各paramごとに、GTを横線として表示する
        for j, param in enumerate(param_list):
            gt_val = gt_data[param]
            # x_positions[j] が param の中心位置
            param_x = x_positions[j]
            halfwidth = total_width / 2.0  # 棒グループの半分くらいの横幅

            ax.hlines(
                y=gt_val,
                xmin=param_x - halfwidth,
                xmax=param_x + halfwidth,
                color='black',
                linestyle='--',
                linewidth=1.0,
                alpha=0.8
            )

        # 軸やラベルの設定
        ax.set_xticks(x_positions)
        ax.set_xticklabels(param_list, rotation=45, ha="right")

        # タイトル(パラメータ群)
        ax.set_title(f"{', '.join(param_list)}", fontsize=12)

        # y 軸ラベル（単位を入れたいなら group_units[param_list[0]] などで可）
        ax.set_ylabel("Value")

        # グリッドを追加
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # 凡例を (一番左サブプロットだけ など必要に応じて)
        if subplot_index == 0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()


def main(gt_path: str, algo_dirs: list):
    """
    GT(JSONファイル)と任意個数のアルゴリズムディレクトリを受け取り、
    カメラパラメータや再投影誤差に対して、元の推定値(平均±標準偏差)を
    エラーバー付き棒グラフで描画し、各パラメータのGT(真値)を
    横線(hline)として表示する。

    さらに、外れ値によってグラフが見づらくならないよう、
    GT と各推定値(±標準偏差) の分布を見て y軸範囲を自動調整します。

    今回は下記3つのグループにまとめて表示する:
      1) ["reproj_error"] (単独)
      2) ["fx", "fy", "cx", "cy"]
      3) ["k1", "k2", "k3", "p1", "p2"]
    """

    # ==========================
    # 1. GT(真値) を読み込み & グループの定義
    # ==========================
    gt_data_ = load_json(gt_path)

    # グラフでまとめるパラメータ群
    param_groups = [
        ["reproj_error"],                # グループ1: 再投影誤差(1つだけ)
        ["fx", "fy", "cx", "cy"],        # グループ2: fx, fy, cx, cy
        ["k1", "k2", "k3", "p1", "p2"]   # グループ3: 歪み係数たち
    ]

    # 真値を取り出し (キーを対応付け)
    gt_data = {}
    gt_data["reproj_error"] = 0.0
    gt_data["fx"] = gt_data_["intrin"]["fx"]
    gt_data["fy"] = gt_data_["intrin"]["fy"]
    gt_data["cx"] = gt_data_["intrin"]["cx"]
    gt_data["cy"] = gt_data_["intrin"]["cy"]
    gt_data["k1"] = gt_data_["intrin"]["k1"]
    gt_data["k2"] = gt_data_["intrin"]["k2"]
    gt_data["k3"] = gt_data_["intrin"]["k3"]
    gt_data["p1"] = gt_data_["intrin"]["p1"]
    gt_data["p2"] = gt_data_["intrin"]["p2"]

    # (オプション) 単位を入れたい場合はここで param→unit のマッピングを定義
    group_units = {
        "reproj_error": "",
        "fx": "",
        "fy": "",
        "cx": "",
        "cy": "",
        "k1": "",
        "k2": "",
        "k3": "",
        "p1": "",
        "p2": ""
    }

    # ==========================
    # 2. アルゴリズム結果の読み込み
    # ==========================
    algo_results = {}

    for algo_dir in algo_dirs:
        algo_name = os.path.basename(algo_dir)

        # 2パターン (_img, _gt) を比較したい想定
        settings = ["_img", "_gt"]
        #settings = ["_gt"]  # 1つだけの場合
        #settings = ["_img"]
        for setting in settings:
            key = algo_name + setting
            algo_results[key] = {
                "reproj_error": [],
                "fx": [], "fy": [], "cx": [], "cy": [],
                "k1": [], "k2": [], "k3": [], "p1": [], "p2": []
            }

            # サブフォルダ( trial_01 など )を走査
            for algo_trial in sorted(Path(algo_dir).iterdir()):
                if not algo_trial.is_dir():
                    continue

                if setting == "_gt":
                    calib_json_path = algo_trial / "calibration.json"
                else:
                    calib_json_path = algo_trial / "detected" / "calibration.json"

                if calib_json_path.exists():
                    data = load_json(calib_json_path)

                    # 必要な値を読み取り
                    reproj_error = float(data["ret"])
                    fx = float(data["mtx"][0][0])
                    fy = float(data["mtx"][1][1])
                    cx = float(data["mtx"][0][2])
                    cy = float(data["mtx"][1][2])
                    k1 = float(data["dist"][0][0])
                    k2 = float(data["dist"][0][1])
                    k3 = float(data["dist"][0][4])
                    p1 = float(data["dist"][0][2])
                    p2 = float(data["dist"][0][3])

                    # 格納
                    algo_results[key]["reproj_error"].append(reproj_error)
                    algo_results[key]["fx"].append(fx)
                    algo_results[key]["fy"].append(fy)
                    algo_results[key]["cx"].append(cx)
                    algo_results[key]["cy"].append(cy)
                    algo_results[key]["k1"].append(k1)
                    algo_results[key]["k2"].append(k2)
                    algo_results[key]["k3"].append(k3)
                    algo_results[key]["p1"].append(p1)
                    algo_results[key]["p2"].append(p2)

    # ==========================
    # 3. グラフ作成の下準備
    # ==========================
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(param_groups),
        figsize=(6 * len(param_groups), 5),
        squeeze=False
    )
    axes = axes[0]

    # アルゴリズムをソートして固定の順序に
    algo_names_sorted = sorted(algo_results.keys())

    # カラーマップ
    cmap = cm.get_cmap("tab10", len(algo_names_sorted))
    colors = [cmap(i) for i in range(len(algo_names_sorted))]

    # 全体タイトル
    #fig.suptitle("Camera Parameters (Original Values) with GT lines", fontsize=16, y=1.02)

    # ==========================
    # 4. グラフ描画: 元の値 & GTライン
    # ==========================
    for subplot_index, param_list in enumerate(param_groups):
        ax = axes[subplot_index]

        x_positions = np.arange(len(param_list))
        total_width = 0.8
        width_per_algo = total_width / len(algo_names_sorted)

        # このサブプロット全体の最小・最大値を追跡する (y軸自動調整用)
        all_min = float('inf')
        all_max = -float('inf')

        # --- 棒グラフの描画 ---
        for algo_i, algo_name in enumerate(algo_names_sorted):
            means = []
            stds = []
            for param in param_list:
                vals = np.array(algo_results[algo_name][param])
                if len(vals) > 0:
                    m = vals.mean()
                    s = vals.std()
                else:
                    m = np.nan
                    s = 0.0
                means.append(m)
                stds.append(s)

            # バーをずらすオフセット
            offset = (algo_i - (len(algo_names_sorted) - 1) / 2) * width_per_algo
            x_for_this_algo = x_positions + offset

            # バー描画 (元の値)
            ax.bar(
                x_for_this_algo,
                means,
                yerr=stds,
                width=width_per_algo * 0.9,
                color=colors[algo_i],
                alpha=0.8,
                capsize=4,
                label=algo_name
            )

            # # バー上の数値
            for x_pos, mean_val in zip(x_for_this_algo, means):
                ax.text(
                    x_pos,
                    mean_val,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

            # y 範囲の最小・最大を更新 (バー + エラーバー)
            # means ± stds の範囲を考慮
            local_min = np.nanmin(np.array(means) - np.array(stds))
            local_max = np.nanmax(np.array(means) + np.array(stds))
            all_min = min(all_min, local_min)
            all_max = max(all_max, local_max)

        # --- GTラインの描画 ---
        max_gt_val = -float('inf')
        min_gt_val = float('inf')
        for j, param in enumerate(param_list):
            gt_val = gt_data[param]
            param_x = x_positions[j]
            halfwidth = total_width / 2.0  # 水平線を描画する横幅

            # 破線として描画 (paramの中央付近に線を引く)
            ax.hlines(
                y=gt_val,
                xmin=param_x - halfwidth,
                xmax=param_x + halfwidth,
                color='black',
                linestyle='--',
                linewidth=1.2,
                alpha=0.8
            )

            max_gt_val = max(max_gt_val, gt_val)
            min_gt_val = min(max_gt_val, gt_val)

            # GT も 最小・最大の候補に含める
            all_min = min(all_min, gt_val)
            all_max = max(all_max, gt_val)

        # --- y軸の調整 ---
        # 外れ値が大きすぎる場合でも、全体をなるべく見やすくする。
        # 例: min, max の 10%余白をとる。
        if all_min < float('inf') and all_max > -float('inf'):
            margin = 0.1 * (all_max - all_min)
            y_low = all_min - margin
            y_high = all_max + margin

            # y_low = max(-np.abs(min_gt_val)- margin, y_low)
            # y_high = min(max_gt_val * 2.0 + margin, y_high)

            # y_low == y_high の極端なケース防止
            if abs(y_high - y_low) < 1e-9:
                y_low -= 0.5
                y_high += 0.5
            ax.set_ylim(y_low, y_high)

            print(gt_val, y_low, y_high)

        # 軸・ラベル設定
        ax.set_xticks(x_positions)
        ax.set_xticklabels(param_list, rotation=45, ha="right")
        ax.set_title(f"{', '.join(param_list)}", fontsize=12)
        ax.set_ylabel("Value")

        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # 凡例 (必要に応じて1回だけ表示も可)
        if subplot_index == 0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()

# ========== 使い方 (例) ==========
if __name__ == "__main__":
    # コマンドライン引数をパースするなどお好みで処理
    # ここでは直接指定の例
    gt_path_example = "./data/settings/base_setting.json"
    algo_dirs_example = ["./render/transnoise"]
    main(gt_path_example, algo_dirs_example)
