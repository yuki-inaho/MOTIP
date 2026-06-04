# 作業計画書 兼 記録書: MOTIP tracklet疑似正解ラベル学習

---

**日付:** `2026年06月04日`
**作業ディレクトリ・リポジトリ:** 主記録 `/home/kasm-user/Desktop/motip_sandbox`、MOTIP本体 `/home/kasm-user/Desktop/MOTIP`、DEIM worktree `/home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training`
**作業者:** `Codex統括`

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

* **目標1:** Desktop以下にDEIM側の独立worktreeを作成し、semi-auto MOT tracklet生成元のDEIM環境を参照可能にする。
* **目標2:** semi-auto MOTで生成済みのtracklet付きCOCOを、MOTIPが読める疑似正解ラベルDatasetへ変換する。
* **目標3:** `/home/kasm-user/Desktop/MOTIP` のuv/cu118環境上で、tracklet疑似正解ラベルを使ったMOTIP学習を実際に起動し、成功ログまたは明確なblockerを記録する。
* **目標4:** `write-workdoc-uv` と `review-written-workdoc` の方針に従い、作業計画・レビュー・実行記録・検証証跡を監査可能に残す。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** tomato semi-auto annotation / MOT工程で得たtrackletを人手ラベルの代わりに使い、MOTIPの学習がこの新規PCのuv/GPU環境で回る状態まで進めたい。
* **明示要求:**
  * `DEIM_sandbox` 側もDesktop以下にworktreeを作る。
  * 必要ならDEIM側・MOTIP側に「いい感じ」の改変を加えてよい。
  * semi-auto MOTで生成したtrackletを疑似正解ラベルとして使う。
  * MOTIPの学習をuv環境上で回す。
  * `write-workdoc-uv` / `review-written-workdoc` スキルを使い、作業書を作成・レビューする。
  * 作業記録をつけながら進める。
* **暗黙制約:**
  * 作業開始前と重要ステップ前に `date "+%Y-%m-%d %H:%M:%S %Z%z"` を実行し、記録へ転記する。
  * Python作業はuv環境で行い、MOTIP本体は `/home/kasm-user/Desktop/MOTIP/.venv` を使う。
  * `justfile` がある場合は、再現しやすいjust targetを優先する。
  * 暗黙fallbackは禁止。入力ファイル、pretrain、GPUメモリ、checksum等が不足する場合は明示的に失敗させ、証跡を残す。
  * generated artifacts、`.venv/`、checkpoint、動画、大規模datasetはgitへ混入させない。
  * DRY/KISS/SOLID、t-wada TDD、監査性を意識し、変換・dataset loader・train configを小さく検証可能に分ける。
* **非ゴール:**
  * tomato pipeline P2-P7の再実行はしない。既存成果物を入力として使う。
  * MOTIPの本格的な長時間収束学習や精度評価は今回の完了条件に含めない。少なくとも学習loopがtracklet疑似正解で動くことを確認する。
  * checkpoint/videoの厳密な外部provenance検証は、必要なら別作業とする。
  * `motip_sandbox` のJun02 mock smokeを正式成果とみなさない。今回の成果はMOTIP本体uv環境での実行証跡で判定する。
* **成功条件:**
  * Desktop以下にDEIM worktreeが存在し、branch/remote/HEADが記録されている。
  * `near_coco.json` または `coco_good.json` から、track_idを保持したMOTChallenge形式のPseudoMOT datasetが生成される。
  * MOTIP本体にPseudoMOT loader/config/変換helper/just targetが入り、単体テストまたはloader smokeが成功する。
  * `just train-tracklet-smoke` などのuv commandでMOTIP training loopが起動し、少なくとも有限lossを出して終了する、またはGPU/pretrain等の明確なblockerを証跡付きで記録する。
  * 作業書のチェックリスト、作業記録、DoDが全て更新され、未完了項目が残っていない。
* **リスクと前提:**
  * `tracking.csv/json` は `tracker_id=-1` を多く含むため、疑似正解には `attributes.track_id` が入った `near_coco.json` を優先する。
  * MOTIP公式trainingは `DETR_PRETRAIN` を要求する。既存のDanceTrack checkpointをpretrainとして流用できない場合、DETR pretrainの取得またはMOTIP初期化方針の明示が必要。
  * L4 23GBでもMOTIP full configは重い。smoke用に画像サイズ・sample length・batch・worker・epochを下げる。
  * `motip_sandbox` にはPseudoMOT patchがあるが、現在のMOTIP `cu118` branchには未反映。直接適用ではなく現行MOTIPの構造に合わせて必要差分だけ取り込む。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 作業書作成・レビュー | 目標4 | 本作業書、レビュー記録 | `rg -n -- "- \\[ \\]"`、review結果 |
| SG-2 | DEIM Desktop worktree整備 | 目標1 | `/home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training` | `git status --short --branch`, `git worktree list` |
| SG-3 | tracklet入力調査 | 目標2 | 入力COCO/画像/track_idの件数記録 | Python/JSON集計、作業記録 |
| SG-4 | PseudoMOT変換実装 | 目標2 | `tools/convert_coco_tracklets_to_pseudomot.py`, `datasets/TomatoTrackletMOT/` | `uv run pytest ...`, `just build-tracklet-pseudomot` |
| SG-5 | MOTIP Dataset/config統合 | 目標3 | `data/pseudo_mot.py`, `data/joint_dataset.py`, tracklet train config | loader smoke / config check |
| SG-6 | uv上のMOTIP学習実行 | 目標3 | training log/checkpointまたはblocker log | `just train-tracklet-smoke`, finite loss確認 |
| SG-7 | 監査・記録 | 目標4 | 作業記録、DoD、git status | workdoc最終確認、必要なら監査エージェント結果 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | DEIM worktreeをDesktop以下へ作る | 手順4 | `git worktree list`, `git status --short --branch` |
| TR-2 | semi-auto MOT trackletを疑似正解に使う | 手順5-8 | `near_coco.json`集計、MOTChallenge `gt.txt` |
| TR-3 | MOTIP uv環境で学習を回す | 手順9-14 | `just env-info`, `just train-tracklet-smoke` |
| TR-4 | write/reviewスキルで作業書を作る | 手順1-3 | 本書、review記録 |
| TR-5 | 暗黙fallback禁止・監査性 | 全手順 | 作業記録、エラー時対処、DoD |
| TR-6 | 大規模生成物をgitに混ぜない | 手順15 | `git check-ignore`, `git ls-files` |

---

## 2. 作業内容

### フェーズ 1: 作業書・前提確認 (見積: 0.5h)

1. **作業書作成:** `write-workdoc-uv` のテンプレートに従い、本書を作成する。
2. **作業書レビュー:** `review-written-workdoc` のrubricに従い、実行可能性とDoDを確認し、安全な改善を反映する。
3. **現状確認:** MOTIP/DEIM/tomato tracking成果物のpath、branch、主要ファイルを記録する。

### フェーズ 2: DEIM worktreeとtracklet入力調査 (見積: 0.5h)

1. **DEIM worktree:** `/workspace/Project/DEIM_sandbox` からDesktop以下に専用worktreeを作成する。
2. **tracklet入力:** `near_coco.json` を第一候補、`coco_good.json` を第二候補として、track_id、画像、bbox、scoreの件数を集計する。
3. **画像対応:** COCO `file_name` が `/home/kasm-user/Desktop/NYX660_2025_12_01_17_33_27_0135/Color/*.jpg` に対応することを確認する。

### フェーズ 3: PseudoMOT変換・MOTIP統合 (見積: 1.5h)

1. **TDD:** 変換helperの期待仕様をテストで固定する。
2. **変換helper:** COCO tracklet annotationsをMOTChallenge 9列 `gt.txt`、`seqinfo.ini`、`img1/*.jpg` へ変換する。
3. **Dataset loader:** MOTIP本体に `PseudoMOT` loaderを追加し、`JointDataset` から選択可能にする。
4. **config/just:** tracklet training smoke用configとjust targetを追加する。

### フェーズ 4: uv学習実行・監査 (見積: 1.0h)

1. **前処理実行:** tracklet PseudoMOT datasetを生成する。
2. **loader smoke:** MOTIP Datasetが生成datasetを読めることを確認する。
3. **training smoke:** uv環境でMOTIPの学習loopを起動し、有限lossまたは明確なblockerを記録する。
4. **監査:** git状態、生成物ignore、作業記録、DoDを確認する。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。各手順の完了直後に「作業記録」へ日時・結果・証跡を追記します。*

### フェーズ 1: 作業書・前提確認

### 手順 1: 現在時刻と作業対象を記録する
- [x] 🖐 **操作**: `date "+%Y-%m-%d %H:%M:%S %Z%z"` を実行し、MOTIP/DEIM/tracking成果物のpathを本書へ記録する。
- [x] 🔎 **確認**: 作業記録に日時、対象repo、入力成果物pathが1行追加されている。
- [x] 🧪 **テスト**: `manual_workdoc_start_record`。作業記録の先頭行にdate出力が含まれることを目視確認する。
- [x] 🛠 **エラー時対処**: `date` 形式やtimezoneが異なる場合は出力そのものを記録し、必要なら `TZ=Asia/Tokyo date ...` を併記する。

### 手順 2: 本作業書を作成する
- [x] 🖐 **操作**: `write-workdoc-uv` テンプレートに沿って `temp/workdoc_Jun04-2026_motip_tracklet_training.md` を作成する。
- [x] 🔎 **確認**: 目的、ゴール要求分析、サブゴール、Trace ID、フェーズ、チェックリスト、DoD、作業記録が存在する。
- [x] 🧪 **テスト**: `rg -n "ゴール要求分析|完了の定義|作業記録" temp/workdoc_Jun04-2026_motip_tracklet_training.md` が該当行を返す。
- [x] 🛠 **エラー時対処**: placeholderが残る場合は作業開始前に具体path・commandへ置換する。

### 手順 3: 本作業書をレビューして改善する
- [x] 🖐 **操作**: `review-written-workdoc` rubricで本書を確認し、Blocker/Majorを修正する。
- [x] 🔎 **確認**: レビュー結果が作業記録にあり、残課題が明示されている。
- [x] 🧪 **テスト**: `rg -n -- "- \\[ \\]"` で意図しない未完了項目以外がないことを確認する。
- [x] 🛠 **エラー時対処**: 判断が分かれる仕様は推測で埋めず、「前提」または「保留」として記録する。

### フェーズ 2: DEIM worktreeとtracklet入力調査

### 手順 4: DEIM worktreeをDesktop以下に作る
- [x] 🖐 **操作**: `/workspace/Project/DEIM_sandbox` から `/home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training` へ、`deimv1+cu128` を起点にした専用branch `motip-tracklet-training` のworktreeを作成する。
- [x] 🔎 **確認**: `git worktree list` と `git status --short --branch` でDesktop worktreeのbranch/HEADが確認できる。
- [x] 🧪 **テスト**: `manual_deim_worktree_exists`。`test -d /home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training/.git` またはworktree gitfileが存在する。
- [x] 🛠 **エラー時対処**: 既にディレクトリがある場合は削除せず、statusを確認して既存worktreeの再利用または別名作成を記録する。

### 手順 5: tracklet付きCOCOを集計する
- [x] 🖐 **操作**: `near_coco.json` と `coco_good.json` のimages/annotations/track_id件数を集計し、第一候補を決める。
- [x] 🔎 **確認**: 採用入力、総frame数、annotation数、unique track_id数、score範囲が記録されている。
- [x] 🧪 **テスト**: `manual_tracklet_coco_has_track_id`。全annotationまたは採用対象annotationが `attributes.track_id` を持つ。
- [x] 🛠 **エラー時対処**: `track_id` が欠損する場合は `tracking.csv/json` を使わず、P2/P2f生成手順へ戻るかblockerとして記録する。

### 手順 6: COCO画像file_nameと実画像を照合する
- [x] 🖐 **操作**: COCO `file_name` と `/home/kasm-user/Desktop/NYX660_2025_12_01_17_33_27_0135/Color` の画像を照合する。
- [x] 🔎 **確認**: 採用COCOの全imageに対応する画像が存在する、または欠損一覧が記録されている。
- [x] 🧪 **テスト**: `manual_coco_images_exist`。欠損数が0であることを確認する。
- [x] 🛠 **エラー時対処**: 欠損がある場合は対象image/annotationを除外するか、入力datasetを切り替える。暗黙に別画像dirを使わない。

### フェーズ 3: PseudoMOT変換・MOTIP統合

### 手順 7: 変換仕様のテストを追加する
- [x] 🖐 **操作**: MOTIP本体に、COCO trackletからMOTChallenge行へ変換する単体テストを追加する。
- [x] 🔎 **確認**: 実装前に対象importまたは関数未定義で失敗する。
- [x] 🧪 **テスト**: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_tracklet_pseudomot_conversion.py -q` がRedを示す。
- [x] 🛠 **エラー時対処**: pytest自体が動かない場合はuv環境とdev dependencyを確認し、依存不足を作業記録に残す。

### 手順 8: COCO trackletをPseudoMOTへ変換するhelperを実装する
- [x] 🖐 **操作**: `tools/convert_coco_tracklets_to_pseudomot.py` を追加し、COCO bbox `[x,y,w,h]` と `attributes.track_id` からMOTChallenge 9列を生成する。
- [x] 🔎 **確認**: `gt.txt`, `seqinfo.ini`, `img1/*.jpg`, `conversion_summary.json` が生成される。
- [x] 🧪 **テスト**: 手順7のテストがGreenになり、`just build-tracklet-pseudomot` が終了コード0で成功する。
- [x] 🛠 **エラー時対処**: 1-index/0-index不整合が出た場合はCOCO `image_id` と `file_name` の対応表を出力し、MOTChallenge frame idを1始まりで固定する。

### 手順 9: MOTIP PseudoMOT loaderを追加する
- [x] 🖐 **操作**: `data/pseudo_mot.py` を追加し、`data/joint_dataset.py` に `PseudoMOT` 登録と必要kwargsを追加する。
- [x] 🔎 **確認**: `JointDataset(DATASETS=["PseudoMOT"])` が生成datasetを読み、`statistics()` が1 sequenceとframe数を返す。
- [x] 🧪 **テスト**: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_pseudo_mot_dataset.py -q` が成功する。
- [x] 🛠 **エラー時対処**: `Dataset PseudoMOT is not supported` の場合は登録漏れを確認する。`is_legal` 失敗時は空frame/重複track_id/300件超過をsummaryで切り分ける。

### 手順 10: tracklet training smoke configを追加する
- [x] 🖐 **操作**: `configs/train_tracklet_pseudomot_smoke.yaml` を追加し、DATA_ROOT/DATASETS/SAMPLE_LENGTHS/EPOCHS/OUTPUTS_DIR/EXP_NAMEをtracklet用に設定する。
- [x] 🔎 **確認**: config load後に `DATASETS: [PseudoMOT]`, `DATASET_SPLITS: [train]`, `EXP_NAME` が明確に表示される。
- [x] 🧪 **テスト**: `just config-tracklet-smoke` がtracklet configの主要キーを出力する。
- [x] 🛠 **エラー時対処**: pretrainが存在しない場合は自動downloadせず、公式URL・不足path・代替案を記録する。

### 手順 11: just targetを追加する
- [x] 🖐 **操作**: `justfile` に `build-tracklet-pseudomot`, `prepare-tracklet-pretrain`, `loader-tracklet-smoke`, `config-tracklet-smoke`, `train-tracklet-smoke` を追加する。
- [x] 🔎 **確認**: `just --list` に5 targetが表示される。
- [x] 🧪 **テスト**: `just --list | rg "tracklet"` が5 targetを返す。
- [x] 🛠 **エラー時対処**: 引数のpathが長い場合は変数化し、Desktop絶対pathを記録する。

### フェーズ 4: uv学習実行・監査

### 手順 12: tracklet PseudoMOT datasetを生成する
- [x] 🖐 **操作**: `just build-tracklet-pseudomot` を実行する。
- [x] 🔎 **確認**: `datasets/TomatoTrackletMOT/train/nyx660_jun04/gt/gt.txt` と `seqinfo.ini` が生成される。
- [x] 🧪 **テスト**: `python3 -m json.tool datasets/TomatoTrackletMOT/conversion_summary.json` が成功し、valid frames/objects/track countが記録されている。
- [x] 🛠 **エラー時対処**: object数が0の場合はscore閾値、input COCO、track_id存在を再確認する。

### 手順 13: PseudoMOT loader smokeを実行する
- [x] 🖐 **操作**: `just loader-tracklet-smoke` を実行する。
- [x] 🔎 **確認**: loaderがsequence数、frame数、sample数を出力し、例外なく終了する。
- [x] 🧪 **テスト**: loader smoke終了コード0。
- [x] 🛠 **エラー時対処**: `trajectory_id_labels` 欠損はMOTIP transform適用順の問題として `data/transforms.py` とsample lengthを確認する。

### 手順 14: MOTIP学習smokeをuv環境で実行する
- [x] 🖐 **操作**: `just train-tracklet-smoke` を実行し、tracklet疑似正解でMOTIP training loopを起動する。
- [x] 🔎 **確認**: finite `loss` / `detr_loss` / 可能なら `id_loss` がlogに出る、またはpretrain/GPU等のblockerが明確に記録される。
- [x] 🧪 **テスト**: `manual_train_tracklet_smoke`。`outputs/tracklet_pseudomot_smoke/train` にlogが生成される。
- [x] 🛠 **エラー時対処**: CUDA OOM時はsample length、AUG_MAX_SIZE、batch、workerを下げて再試行する。pretrain不足時は不足pathと取得候補を記録する。

### 手順 15: 差分と生成物ignoreを監査する
- [x] 🖐 **操作**: `git status --short --branch`, `git ls-files`, `git check-ignore -v` でMOTIP/DEIM/sandboxの状態を確認する。
- [x] 🔎 **確認**: commit候補とgenerated artifactが分離されている。
- [x] 🧪 **テスト**: `.venv/`, `datasets/TomatoTrackletMOT/`, `outputs/`, checkpointがtrackedされていない。
- [x] 🛠 **エラー時対処**: 大容量生成物がtracked候補に入った場合は `.gitignore` を修正し、stage前に必ず再確認する。

### 手順 16: 完了の定義と作業記録を最終更新する
- [x] 🖐 **操作**: DoDを確認し、成功/未完/制約事項を作業記録へ追記する。
- [x] 🔎 **確認**: 本書に未完了チェックがなく、最終状態表にpath/branch/commit/log/制約がある。
- [x] 🧪 **テスト**: `rg -n -- "- \\[ \\]" temp/workdoc_Jun04-2026_motip_tracklet_training.md` が意図しない未完了を返さない。
- [x] 🛠 **エラー時対処**: 未完了が残る場合は完了扱いにせず、blockerと次アクションを明記する。

---

## 4. 作業に使用するコマンド参考情報

```bash
date "+%Y-%m-%d %H:%M:%S %Z%z"

git -C /workspace/Project/DEIM_sandbox worktree list
git -C /workspace/Project/DEIM_sandbox worktree add -b motip-tracklet-training /home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training deimv1+cu128

cd /home/kasm-user/Desktop/MOTIP
just env-info
just build-tracklet-pseudomot
just loader-tracklet-smoke
just train-tracklet-smoke
UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py -q
UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run ruff check tools/convert_coco_tracklets_to_pseudomot.py data/pseudo_mot.py tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py
```

---

## 5. 完了の定義

- [x] DoD-01: 本作業書が `temp/workdoc_Jun04-2026_motip_tracklet_training.md` に存在し、review結果が記録されている。
- [x] DoD-02: `/home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training` が存在し、branch/HEAD/remoteが記録されている。
- [x] DoD-03: 採用tracklet COCOのimages/annotations/unique track_id/画像欠損数が記録されている。
- [x] DoD-04: `tools/convert_coco_tracklets_to_pseudomot.py` とテストが存在し、tracklet COCOからMOTChallenge形式へ変換できる。
- [x] DoD-05: MOTIP本体が `PseudoMOT` datasetを読み込める。
- [x] DoD-06: tracklet training smoke configとjust targetが存在する。
- [x] DoD-07: uv環境で `just train-tracklet-smoke` を実行し、finite lossまたは明確なblockerを記録している。
- [x] DoD-08: `.venv/`, `outputs/`, `datasets/TomatoTrackletMOT/`, checkpoint等のgenerated artifactsがgit管理対象外である。
- [x] DoD-09: 作業記録に実行コマンド、成功/失敗、修正内容、制約事項、次アクションが記録されている。

---

## 6. 作業記録

作業記録上の注意:

- 記録は推測で補完せず、実行したコマンド・観測した出力・編集したファイルに基づいて書く。
- 失敗や制約は隠さず、再現条件と次アクションを併記する。
- チェックリストを `[x]` にした直後に、その手順の結果をこの表へ追記する。
- generated artifactsとcommit対象の区別を毎回明示する。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `2026-06-04` | `04:05:03 UTC+0000` | `Codex統括` | 作業開始・前提確認 | ✅開始: `write-workdoc-uv` / `review-written-workdoc` のskill本文を確認。MOTIPは `/home/kasm-user/Desktop/MOTIP` の `cu118...origin/cu118`、tomato tracking成果物は `/home/kasm-user/Desktop/tomato_tracking_deim_mot/outputs/nyx660_jun04/{tracking.csv,tracking.json,coco_good.json,near_coco.json}`。 |
| `2026-06-04` | `04:05:03 UTC+0000` | `Codex統括` | 手順1完了: 現在時刻と対象path記録 | ✅成功: date出力 `2026-06-04 04:05:03 UTC+0000`、MOTIP repo、DEIM元repo、tracklet成果物pathを本書へ反映。 |
| `2026-06-04` | `04:05:03 UTC+0000` | `Codex統括` | 手順2完了: 作業書作成 | ✅成功: `temp/workdoc_Jun04-2026_motip_tracklet_training.md` を作成。ゴール要求分析、サブゴール、Trace ID、phase、16手順チェックリスト、DoD、作業記録を含む。 |
| `2026-06-04` | `04:09:41 UTC+0000` | `Codex統括` | 手順3レビュー途中: worktreeコマンド修正 | ✅改善: `review-written-workdoc` rubricで確認し、手順4の当初コマンド `git worktree add ... deimv1+cu128` は元worktreeで同branchがcheckout中のため失敗し得ると判断。専用branch `motip-tracklet-training` を `deimv1+cu128` から作る形に修正。`git worktree list` は `/workspace/Project/DEIM_sandbox 0124409 [deimv1+cu128]`、`motip-tracklet-training` branchは未作成。 |
| `2026-06-04` | `04:09:41 UTC+0000` | `Codex統括` | 手順3完了: 作業書レビュー | ✅PASS_WITH_NOTES: Blocker/Majorは修正済み。残る注意は、`just train-tracklet-smoke` がpretrain/GPU制約で停止する可能性がある点で、本書では明確なblocker記録を許容する完了条件にしている。DoD-01を満足。 |
| `2026-06-04` | `04:10:52 UTC+0000` | `Codex統括` | 手順4完了: DEIM Desktop worktree作成 | ✅成功: `git worktree add -b motip-tracklet-training /home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training deimv1+cu128` を実行。`git worktree list` は `/workspace/Project/DEIM_sandbox 0124409 [deimv1+cu128]` と `/home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training 0124409 [motip-tracklet-training]`。Desktop worktreeは `## motip-tracklet-training`, HEAD `0124409 Add LLM onboarding summary`, remote `https://github.com/yuki-inaho/DEIM_sandbox.git`。DoD-02を満足。 |
| `2026-06-04` | `04:11:33 UTC+0000` | `Codex統括` | 手順5完了: tracklet付きCOCO集計と採用入力決定 | ✅成功: `coco_good.json` は images `1219`, annotations `34101`, missing_track_id `0`, unique_track_id `1367`, track_len min/max `(8, 111)`, score min/max `(0.3001, 0.8895)`, objects/frame `(2, 63)`, images_without_ann `0`。`near_coco.json` は images `1219`, annotations `28483`, missing_track_id `0`, unique_track_id `1315`, track_len min/max `(1, 111)`, images_without_ann `23`。MOTIP ID学習では連続trackletと非空frameが有利なため、採用入力は `/home/kasm-user/Desktop/tomato_tracking_deim_mot/outputs/nyx660_jun04/coco_good.json` とする。MOTIP transformはclip内IDを `NUM_ID_VOCABULARY` へ再ラベルするため、全体unique track_idが1367でもsmoke学習可能。 |
| `2026-06-04` | `04:12:30 UTC+0000` | `Codex統括` | 手順6完了: COCO画像照合 | ✅成功: 採用COCO `coco_good.json` の `images` 1219件について `/home/kasm-user/Desktop/NYX660_2025_12_01_17_33_27_0135/Color/{file_name}` を照合し、missing `0`。先頭画像 `/home/kasm-user/Desktop/NYX660_2025_12_01_17_33_27_0135/Color/000002.jpg` はJPEG 800x600でCOCO metadata (`width: 800`, `height: 600`) と一致。DoD-03を満たす。 |
| `2026-06-04` | `04:13:05 UTC+0000` | `Codex統括` | 手順7完了: 変換仕様テスト追加とRed確認 | ✅Red確認: `/home/kasm-user/Desktop/MOTIP/tests/test_tracklet_pseudomot_conversion.py` を追加。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_tracklet_pseudomot_conversion.py -q` は `ModuleNotFoundError: No module named 'tools'` で失敗し、未実装状態を確認。次は `tools/convert_coco_tracklets_to_pseudomot.py` を実装してGreenへ変える。 |
| `2026-06-04` | `04:16:07 UTC+0000` | `Codex統括` | 手順8完了: COCO tracklet→PseudoMOT変換実装 | ✅成功: `/home/kasm-user/Desktop/MOTIP/tools/convert_coco_tracklets_to_pseudomot.py` と `tools/__init__.py` を追加し、`pyproject.toml` にpytest `pythonpath = ["."]` を追加。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_tracklet_pseudomot_conversion.py -q` は `1 passed`。`just build-tracklet-pseudomot` は採用COCOを `datasets/TomatoTrackletMOT` へ変換し、summaryは frames `1219`, objects `34101`, tracks `1367`, min/max frame `1/1219`。`gt.txt` は1.7M、画像は `img1/*.jpg` symlink 1219件、先頭 `00000001.jpg -> .../Color/000002.jpg`。DoD-04を満足。 |
| `2026-06-04` | `04:18:21 UTC+0000` | `Codex統括` | 手順9完了: MOTIP PseudoMOT loader追加 | ✅成功: `data/pseudo_mot.py` を追加し、`data/joint_dataset.py` に `PseudoMOT` 登録、`data/__init__.py` に `PSEUDOMOT_*` config引数伝播を追加。TDDでは `tests/test_pseudo_mot_dataset.py` を先に追加し `ModuleNotFoundError: No module named 'data.pseudo_mot'` を確認後、実装。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py -q` は `2 passed in 1.89s`。実データloader smokeは手順13で実施する。 |
| `2026-06-04` | `04:21:24 UTC+0000` | `Codex統括` | 手順10完了: tracklet training smoke config追加 | ✅成功: `configs/train_tracklet_pseudomot_smoke.yaml` を追加。`just config-tracklet-smoke` は `DATASETS=['PseudoMOT']`, `DATASET_SPLITS=['train']`, `PSEUDOMOT_SUB_DIR=TomatoTrackletMOT`, `SAMPLE_LENGTHS=[2]`, `EPOCHS=1`, `MAX_TRAIN_STEPS=2`, `DETR_PRETRAIN=./pretrains/r50_deformable_detr_coco_dancetrack.pth`, `OUTPUTS_DIR=./outputs/tracklet_pseudomot_smoke`, `EXP_NAME=tracklet_pseudomot_smoke` を表示。 |
| `2026-06-04` | `04:21:24 UTC+0000` | `Codex統括` | 手順11完了: tracklet just target追加 | ✅成功: `justfile` に `build-tracklet-pseudomot`, `prepare-tracklet-pretrain`, `config-tracklet-smoke`, `loader-tracklet-smoke`, `train-tracklet-smoke` を追加。`just --list | rg "tracklet"` で5 targetを確認。DoD-06を満足。 |
| `2026-06-04` | `04:22:09 UTC+0000` | `Codex統括` | 手順12完了: tracklet PseudoMOT dataset生成確認 | ✅成功: `datasets/TomatoTrackletMOT/conversion_summary.json` は JSON妥当。summaryは frames `1219`, objects `34101`, tracks `1367`, min/max frame `1/1219`。`datasets/TomatoTrackletMOT/train/nyx660_jun04/gt/gt.txt` と `seqinfo.ini` が存在。 |
| `2026-06-04` | `04:22:35 UTC+0000` | `Codex統括` | 手順13完了: 実データloader smoke | ✅成功: `just loader-tracklet-smoke` は終了コード0。出力は `['PseudoMOT.train, 1 sequences, 1219 frames.']` と `samples 1218`。`JointDataset` が `TomatoTrackletMOT` を読み、sample length 2の候補を生成できることを確認。DoD-05を満足。 |
| `2026-06-04` | `04:23:28 UTC+0000` | `Codex統括` | 手順14完了: MOTIP tracklet疑似正解学習smoke | ✅成功: 事前に `just prepare-tracklet-pretrain` で既存DanceTrack MOTIP checkpointからDETR keys `597` 件を抽出し、`pretrains/r50_deformable_detr_coco_dancetrack.pth` (156M) を生成。`just train-tracklet-smoke` は終了コード0。`PseudoMOT.train, 1 sequences, 1219 frames.` を読み込み、`MAX_TRAIN_STEPS=2` で早期停止。logには有限値 `loss=43.1608`, `detr_loss=38.4402`, `id_loss=4.7205`, `max_cuda_mem(MB)=1069.8433`。出力は `outputs/tracklet_pseudomot_smoke/checkpoint_0.pth` (約677M), `outputs/tracklet_pseudomot_smoke/train/config.yaml`, `outputs/tracklet_pseudomot_smoke/train/log.txt`。実行後GPU compute processなし。DoD-07を満足。 |
| `2026-06-04` | `04:24:05 UTC+0000` | `Codex統括` | 手順15完了: git差分と生成物ignore監査 | ✅成功: MOTIPは `## cu118...origin/cu118` で実装差分あり。変更候補は `data/__init__.py`, `data/joint_dataset.py`, `justfile`, `pyproject.toml`, `train.py`, 新規 `configs/train_tracklet_pseudomot_smoke.yaml`, `data/pseudo_mot.py`, `tests/`, `tools/__init__.py`, `tools/convert_coco_tracklets_to_pseudomot.py`, `tools/extract_detr_pretrain.py`。`git ls-files datasets/TomatoTrackletMOT outputs/tracklet_pseudomot_smoke pretrains/r50_deformable_detr_coco_dancetrack.pth` は空。`git check-ignore -v` は `datasets/`, `outputs/`, `pretrains/`, `.venv/`, `.ruff_cache/` を `.gitignore` 対象として確認。DEIM Desktop worktreeは `## motip-tracklet-training` clean。DoD-08を満足。 |
| `2026-06-04` | `04:25:26 UTC+0000` | `Codex統括` | 品質確認: unit tests / lint / status | ✅成功: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py -q` は `2 passed in 1.99s`。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run ruff check tools/convert_coco_tracklets_to_pseudomot.py tools/extract_detr_pretrain.py data/pseudo_mot.py tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py train.py` は `All checks passed!`。MOTIP実装差分は未commit、DEIM worktreeはclean。 |
| `2026-06-04` | `04:26:56 UTC+0000` | `Codex統括` | 手順16完了: DoD/最終状態更新 | ✅成功: DoD-01〜DoD-09を確認し、本章の最終状態表へpath/branch/command/log/制約/次アクションを記録。制約は「smoke学習は2 stepであり、本格学習ではepoch/step/validation設計が別途必要」「MOTIP実装差分は未commit」。次アクションは必要に応じたcommit/pushと本格学習条件の調整。 |
| `2026-06-04` | `06:00:36 UTC+0000` | `Codex統括` | 追加対応: MOTIP commit/push | ✅成功: `/home/kasm-user/Desktop/MOTIP` の `cu118` で `Add tracklet pseudomot smoke training` をcommit。commit `241f41d`。`git push origin cu118` は `c5673e1..241f41d cu118 -> cu118`。push後 `git rev-list --left-right --count HEAD...@{u}` は `0 0` で、`HEAD == origin/cu118`。事前確認としてpytestは `2 passed in 1.72s`、ruffは `All checks passed!`。 |

---

## 7. 最終状態

| 項目 | 最終状態 |
| :--- | :--- |
| 作業書 | `/home/kasm-user/Desktop/motip_sandbox/temp/workdoc_Jun04-2026_motip_tracklet_training.md` |
| MOTIP repo | `/home/kasm-user/Desktop/MOTIP`, branch `cu118`, HEAD `241f41d`, `origin/cu118` と同期済み |
| DEIM worktree | `/home/kasm-user/Desktop/DEIM_sandbox_motip_tracklet_training`, branch `motip-tracklet-training`, HEAD `0124409`, status clean |
| 採用疑似正解 | `/home/kasm-user/Desktop/tomato_tracking_deim_mot/outputs/nyx660_jun04/coco_good.json`。images `1219`, annotations `34101`, unique track_id `1367`, missing image `0` |
| PseudoMOT dataset | `/home/kasm-user/Desktop/MOTIP/datasets/TomatoTrackletMOT`。frames `1219`, objects `34101`, tracks `1367`, `img1/*.jpg` はNYX660 Colorへのsymlink |
| 追加実装 | COCO tracklet -> MOTChallenge変換、`PseudoMOT` dataset loader、tracklet smoke config、just target、`MAX_TRAIN_STEPS` 早期停止、DETR pretrain抽出 |
| 学習smoke | `just train-tracklet-smoke` 成功。`MAX_TRAIN_STEPS=2` で終了し、finite `loss=43.1608`, `detr_loss=38.4402`, `id_loss=4.7205`, `max_cuda_mem(MB)=1069.8433` |
| 学習ログ/出力 | `/home/kasm-user/Desktop/MOTIP/outputs/tracklet_pseudomot_smoke/train/log.txt`, `config.yaml`, `checkpoint_0.pth` |
| 検証 | `pytest tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py -q` -> `2 passed in 1.99s`。`ruff check ... train.py` -> `All checks passed!` |
| generated artifacts | `.venv/`, `datasets/TomatoTrackletMOT/`, `outputs/tracklet_pseudomot_smoke/`, `pretrains/r50_deformable_detr_coco_dancetrack.pth`, caches はgit管理対象外 |
| 制約 | 今回はuv環境でMOTIP training loopが疑似正解trackletを読んで回ることを確認する2-step smoke。本格学習のepoch数、validation、評価指標、checkpoint保存方針は別途調整する。 |
| 次アクション | 本格学習へ進む場合は `MAX_TRAIN_STEPS` を外し、`NUM_ID_VOCABULARY` / `NUM_TRAINING_IDS` / augmentation / validation splitを設計する。 |
