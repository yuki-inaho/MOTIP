# 作業計画書 兼 記録書: MOTIP 5FPSサブサンプリングID fine-tune

---

**日付:** `2026年06月04日`
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/MOTIP`（Git repository, branch `cu118`, remote `git@github.com:yuki-inaho/MOTIP.git`, uv venv `/home/kasm-user/Desktop/MOTIP/.venv`）
**作業者:** `Codex統括`

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

* **目標1:** 30FPS疑似正解trackletを5FPS相当へサブサンプリングしたPseudoMOT datasetを作成する。
* **目標2:** MOTIPのID一貫性を改善するため、5FPS相当datasetで `SAMPLE_LENGTHS=[16]` のID強化fine-tuneを実行する。
* **目標3:** 新checkpointで全1219フレームの推論JSON/MOT txtと3fps可視化mp4を生成し、旧モデルとID proxy metricsを比較する。
* **目標4:** 作業書・テスト・DoD・commit/pushまで監査可能に完了させる。
* **目標5:** MOTIP推論結果の `track_id` が毎フレーム変わる問題に対し、推論bbox列を検出入力として後段trackerで再ID付けし、画面下から上へ伸びるtracklet候補を生成する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** 現在のMOTIPモデルは検出bboxは良いがIDが毎フレーム新規化しやすい。元データは30FPSなので、tracklet疑似正解を5FPS相当に間引いて、同じclip長でも実時間の関連付け文脈を伸ばし、ID一貫性を改善したい。
* **明示要求:**
  * 「Implement the plan」: 直前の提案計画を実装する。
  * DoDを明確に定義し、満たすまで作業する。
  * 30FPSから5FPS相当へサブサンプリングして学習してよい。
  * 長時間（最大10時間程度）かけてよい。
  * 作業開始前・重要操作前に `date "+%Y-%m-%d %H:%M:%S %Z%z"` を確認し、作業記録へ残す。
* **暗黙制約:**
  * Python実行はuv環境を使う。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv` と `uv run --no-sync` を基本とする。
  * `justfile` targetを優先し、再現可能なコマンドにまとめる。
  * 暗黙fallback禁止。依存・入力・pretrain・GPU・ID語彙不足は明示失敗させる。
  * `.venv/`, `datasets/`, `outputs/`, `pretrains/`, `*.pth`, `*.mp4`, TensorBoard eventはgitへ混入させない。
  * `AMP_DTYPE=bf16` を使う。L4ではfp16のDETR勾配がoverflowすることが既知。
  * `uv sync --frozen` 後はCUDA opが消えるため、必要時は `just build-ops` し、学習前にop importを確認する。
* **非ゴール:**
  * HOTA/MOTAなど正式なval評価の新規実装はしない。今回は推論結果JSONからのID proxy metricsで比較する。
  * 旧 `outputs/tracklet_pseudomot_full/checkpoint_7.pth` と既存推論成果物は削除しない。
  * MOTIP本体の大規模アーキテクチャ変更はしない。
* **成功条件:**
  * `--frame-stride 6` 対応converterがテストで保護され、既定stride=1の既存挙動が維持される。
  * `datasets/TomatoTrackletMOT_5fps/train/nyx660_jun04_stride6/` が生成され、約204 frames、空frame 0、約5764 objects、`frameRate=5` が記録される。
  * 5FPS fine-tune configが `NUM_ID_VOCABULARY=NUM_TRAINING_IDS=224` 以上でID切詰めを避け、60step preflightと本学習がfinite lossで成功する。
  * 新checkpointで全1219 framesへ推論し、JSON/MOT txt/mp4を生成する。
  * 旧モデルと新モデルの `unique_track_ids / detections`, mean/median track lengthを比較し、改善/未改善を正直に記録する。
  * 後段trackerはMOTIP出力のIDを使わず、bbox/score/categoryのみから再ID付けする。raw推論より `unique_track_ids / detections` を下げ、長いtrackおよびbottom-to-top候補数をsummaryに残す。
  * pytest/ruffがgreen、生成物混入なし、commit/push後に `HEAD == origin/cu118`。
* **リスクと前提:**
  * 5FPS stride=6では `SAMPLE_LENGTHS=[16]` のclip内最大uniqueが203（読取解析）で、vocab 224が必要。
  * stride=6後は204 framesしかないため、1 epochあたりsample数は小さい。10時間枠ではepoch数を多めに取り、checkpoint保存間隔を広げる。
  * 学習のID改善は保証できない。DoDは「改善未達なら未達と記録する」ことを含む。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 作業書・現状確定 | 目標4 | 本作業書、開始時刻、repo/GPU状態 | `date`, `git status`, `nvidia-smi` |
| SG-2 | converter stride対応 | 目標1 | `--frame-stride`, summary拡張、テスト | pytest 赤→緑 |
| SG-3 | 5FPS dataset生成 | 目標1 | `datasets/TomatoTrackletMOT_5fps/` | summary, seqinfo, loader smoke |
| SG-4 | 5FPS ID fine-tune config/just | 目標2 | config, just targets | config dump, vocab計測 |
| SG-5 | preflightと本学習 | 目標2 | log, TB, checkpoint | 60step preflight、本学習exit 0 |
| SG-6 | 推論・動画・比較 | 目標3 | tracks.json, tracks_mot.txt, tracks.mp4, comparison json | proxy metrics比較 |
| SG-7 | 検証・commit/push | 目標4 | pytest/ruff結果、git clean、push | `HEAD == origin/cu118` |
| SG-8 | 後段tracker再ID付け | 目標5 | retrack JSON/MOT txt/summary/mp4, just target, tests | raw比ID proxy改善、動画metadata、pytest/ruff |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | 5FPS相当サブサンプリング | 手順3-5 | converter tests, conversion_summary.json |
| TR-2 | ID制約強化fine-tune | 手順6-9 | config, preflight log, training log |
| TR-3 | 推論JSONと動画生成 | 手順10-11 | infer JSON/MOT txt/mp4 |
| TR-4 | 改善判定 | 手順12 | comparison metrics |
| TR-5 | no-silent-fallback / git hygiene | 全手順 | exceptions, tests, git status/check-ignore |
| TR-6 | 作業記録とDoD | 全手順 | 本書チェックリスト、作業記録、DoD |
| TR-7 | 後段trackerによる再ID付け | 手順16-20 | `tools/retrack_detections.py`, summary, mp4, docs/reports |

---

## 2. 作業内容

### フェーズ 1: 作業書・現状確認

開始時刻、repo状態、GPU状態、既存成果物を確認し、作業書を正本として作成する。

### フェーズ 2: 5FPS dataset対応実装

converterに `frame_stride` を追加し、テストでframe再採番・annotation除外・summary保存を固定する。生成datasetは既存 `TomatoTrackletMOT` とは別名にする。

### フェーズ 3: 5FPS fine-tune設定とpreflight

5FPS dataset用configとjust targetを追加し、clip内最大uniqueを再計測したうえで60step preflightを実行する。

### フェーズ 4: 本学習・推論・動画化

最大10時間枠で本学習を実行し、最終checkpointで全1219 frames推論・3fps mp4生成を行う。

### フェーズ 5: 比較・検証・commit/push

旧モデルと新モデルのID proxy metricsを比較し、pytest/ruff/git hygieneを確認してcommit/pushする。

### フェーズ 6: 後段trackerでbbox列を再ID付け

MOTIPの推論IDは使わず、`tracks.json` のbbox/scoreを検出列として扱い、ByteTrack/OC-SORT相当の後段trackerでIDを付け直す。複数パラメータを試し、raw推論よりID断片化が減ること、長いtrack/bottom-to-top候補が出ること、JSON/MOT txt/summary/mp4を再現可能に生成できることを確認する。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。各手順の完了直後に、作業記録へ日時・結果・証跡を追記します。*

### フェーズ 1: 作業書・現状確認

### 手順 1: 開始時刻と現状を記録する
- [x] 🖐 **操作**: `date "+%Y-%m-%d %H:%M:%S %Z%z"`, `git status --short --branch`, `nvidia-smi` を確認し、本書を作成する。
- [x] 🔎 **確認**: 開始時刻、repo branch、GPU、既存pretrain/checkpointが作業記録にある。
- [x] 🧪 **テスト**: `manual_start_record`。本書が `temp/workdoc_Jun04-2026_motip_5fps_id_finetune.md` に存在する。
- [x] 🛠 **エラー時対処**: repoがdirtyなら差分を読んでから触る。GPU使用中なら学習を開始せず、使用プロセスを記録する。

### フェーズ 2: 5FPS dataset対応実装

### 手順 2: stride変換テストを追加する
- [x] 🖐 **操作**: `tests/test_tracklet_pseudomot_conversion.py` に `frame_stride=2` の小規模COCO変換テストを追加する。
- [x] 🔎 **確認**: 実装前は `TypeError` またはsummary key不足で失敗する。
- [x] 🧪 **テスト**: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run --no-sync pytest tests/test_tracklet_pseudomot_conversion.py -q` の赤→緑。
- [x] 🛠 **エラー時対処**: 期待frame再採番が曖昧なら、source frame idとoutput frame idのmappingをsummaryに追加して期待値を固定する。

### 手順 3: converterに `frame_stride` を実装する
- [x] 🖐 **操作**: `tools/convert_coco_tracklets_to_pseudomot.py` に `frame_stride` 引数とCLI `--frame-stride` を追加し、採用frameだけを1始まりで再採番する。
- [x] 🔎 **確認**: 既定 `frame_stride=1` の既存テストが不変で、stride時は対象外annotationがsummaryで数えられる。
- [x] 🧪 **テスト**: `test_convert_coco_tracklets_to_pseudomot_writes_mot_challenge_files` と新規stride testが成功する。
- [x] 🛠 **エラー時対処**: bbox退化skipとstride skipのaccountingが混ざる場合は、summary keyを分けて不変条件を個別に記録する。

### 手順 4: 5FPS dataset生成just targetを追加する
- [x] 🖐 **操作**: `justfile` に `build-tracklet-pseudomot-5fps`, `loader-tracklet-5fps` を追加する。
- [x] 🔎 **確認**: `just --list` に新targetが表示され、出力先が `datasets/TomatoTrackletMOT_5fps` である。
- [x] 🧪 **テスト**: `just --list | rg "5fps|tracklet"` が新targetを返す。
- [x] 🛠 **エラー時対処**: target引数の順序が紛らわしい場合は変数化し、絶対pathをjustfile冒頭に置く。

### 手順 5: 5FPS datasetを生成して検証する
- [x] 🖐 **操作**: `just build-tracklet-pseudomot-5fps` と `just loader-tracklet-5fps` を実行する。
- [x] 🔎 **確認**: summaryが約204 frames、空frame 0、約5764 objects、`frameRate=5` を示す。
- [x] 🧪 **テスト**: `manual_5fps_dataset_smoke`。loaderが `PseudoMOT.train, 1 sequences, 204 frames.` 相当を返す。
- [x] 🛠 **エラー時対処**: 空frameが出る場合は `PSEUDOMOT_ALLOW_EMPTY_FRAMES` を安易にTrueにせず、stride開始位置または入力COCOを検証する。

### フェーズ 3: 5FPS fine-tune設定とpreflight

### 手順 6: 5FPS fine-tune configを追加する
- [x] 🖐 **操作**: `configs/finetune_tracklet_pseudomot_5fps_id16.yaml` を追加する。
- [x] 🔎 **確認**: `PSEUDOMOT_SUB_DIR=TomatoTrackletMOT_5fps`, `SAMPLE_LENGTHS=[16]`, `SAMPLE_INTERVALS=[1]`, `NUM_ID_VOCABULARY=224`, `AMP_DTYPE=bf16`, `EPOCHS=40`, `OUTPUTS_DIR=./outputs/tracklet_pseudomot_5fps_id16` が設定されている。
- [x] 🧪 **テスト**: config loadが成功し、主要キーdumpで値を確認できる。
- [x] 🛠 **エラー時対処**: vocab不足が見えた場合は実測値+marginへ上げる。fp16は使わない。

### 手順 7: 5FPS学習・推論・動画just targetを追加する
- [x] 🖐 **操作**: `justfile` に `config-tracklet-5fps`, `train-tracklet-5fps`, `infer-tracklet-5fps`, `video-tracklet-5fps` を追加する。
- [x] 🔎 **確認**: 推論targetは新checkpoint・旧30FPS画像列の両方を明示し、出力先は `outputs/tracklet_pseudomot_5fps_id16/infer_30fps/` である。
- [x] 🧪 **テスト**: `just config-tracklet-5fps` が主要キーを表示する。
- [x] 🛠 **エラー時対処**: 推論対象checkpointが未生成ならtargetは明示的に失敗し、旧checkpointへ黙ってfallbackしない。

### 手順 8: 60step preflightを実行する
- [x] 🖐 **操作**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 付きで5FPS config + `MAX_TRAIN_STEPS:60` の一時configを実行する。
- [x] 🔎 **確認**: finite loss、OOMなし、peak CUDA memory < 16GB、ID切詰めなし、EMA/TB/checkpoint正常。
- [x] 🧪 **テスト**: `manual_5fps_preflight`。logとcheckpoint/TB eventを確認する。
- [x] 🛠 **エラー時対処**: OOMなら `AUG_NUM_GROUPS=2` へ下げて再preflight。それでも失敗なら停止しblocker記録する。

### フェーズ 4: 本学習・推論・動画化

### 手順 9: 5FPS本学習を実行する
- [x] 🖐 **操作**: preflight成功後、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True just train-tracklet-5fps` 相当を実行する。
- [x] 🔎 **確認**: 本学習ログを集計し、saturation判定で採用checkpoint、log、TensorBoard eventが残ることを確認する。
- [x] 🧪 **テスト**: `manual_5fps_full_train`。採用loss/id_loss/global_step/checkpoint pathを記録する。
- [x] 🛠 **エラー時対処**: 10時間枠を超える、またはlossがサチった場合は最良正常checkpointを使って評価へ進むことを記録し、未検証checkpointを完走扱いしない。

### 手順 10: 新checkpointで全1219フレーム推論する
- [x] 🖐 **操作**: `just infer-tracklet-5fps` を実行し、旧30FPS全frame列へ推論する。
- [x] 🔎 **確認**: JSON/MOT txtが `outputs/tracklet_pseudomot_5fps_id16/infer_30fps/` に生成される。
- [x] 🧪 **テスト**: `manual_5fps_infer_full`。`num_frames=1219`, `num_detections`, `num_unique_track_ids` を記録する。
- [x] 🛠 **エラー時対処**: checkpoint config不一致は明示失敗させ、config/checkpoint/vocabの対応を確認する。

### 手順 11: 新推論結果から3fps mp4を生成する
- [x] 🖐 **操作**: `just video-tracklet-5fps 3 0` を実行する。
- [x] 🔎 **確認**: `tracks.mp4` が生成され、frame count/fps/sizeを確認する。
- [x] 🧪 **テスト**: `manual_5fps_video`。OpenCVでvideo metadataを読み、3fpsとframe数を記録する。
- [x] 🛠 **エラー時対処**: mp4生成失敗時はJSONとimage pathの対応を確認し、frame dumpは必要時のみ生成する。

### 手順 12: 旧モデルと新モデルのID proxy metricsを比較する
- [x] 🖐 **操作**: 旧 `outputs/tracklet_pseudomot_full/infer/tracks.json` と新 `outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.json` を比較するscriptを追加または一時実行し、summary JSONを保存する。
- [x] 🔎 **確認**: `unique_track_ids / detections`, mean/median track length, detections/frameが旧新で記録される。
- [x] 🧪 **テスト**: `manual_id_proxy_compare`。改善/未改善を数値で判定し、本書へ転記する。
- [x] 🛠 **エラー時対処**: 新モデルが改善しない場合は「改善未達」と明記し、clip長/epoch/LR/thresholdの次アクションを記録する。

### フェーズ 5: 検証・commit/push

### 手順 13: pytestとruffを実行する
- [x] 🖐 **操作**: `uv run --no-sync pytest tests/ -q` と変更ファイル対象の `ruff check` を実行する。
- [x] 🔎 **確認**: pytest passed、ruff All checks passed。
- [x] 🧪 **テスト**: `quality_gate`。passed数とruff結果を作業記録へ転記する。
- [x] 🛠 **エラー時対処**: 失敗したテスト/ruffを修正して再実行し、失敗ログと修正を記録する。

### 手順 14: 生成物混入を監査する
- [x] 🖐 **操作**: `git status`, `git ls-files`, `git check-ignore -v` でdatasets/outputs/pretrains/checkpoint/mp4がtrackedされていないことを確認する。
- [x] 🔎 **確認**: commit対象はsource/config/test/doc/justfileのみで、生成物はgit外。
- [x] 🧪 **テスト**: `git_hygiene_gate`。生成物grepが空であること。
- [x] 🛠 **エラー時対処**: 生成物がstage候補に入った場合はunstageし、必要なら `.gitignore` を修正する。

### 手順 15: commit & pushしDoDを閉じる
- [x] 🖐 **操作**: 必要ファイルだけをstageし、commit後 `git push origin cu118` する。
- [x] 🔎 **確認**: `git rev-list --left-right --count HEAD...@{u}` が `0 0`、working tree clean。
- [x] 🧪 **テスト**: `commit_push_gate`。HEAD hash、commit message、push結果を記録する。
- [x] 🛠 **エラー時対処**: push失敗はremote/auth/branch protectionを確認し、生成物を含めたまま再pushしない。

### フェーズ 6: 後段tracker再ID付け

### 手順 16: raw推論JSONを検出列として分析する
- [x] 🖐 **操作**: `outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.json` を読み、frame数、検出数、score分布、y位置分布を確認する。
- [x] 🔎 **確認**: rawの `track_id` は信頼せず、bbox/score/categoryのみを後段tracker入力とする。
- [x] 🧪 **テスト**: `manual_raw_detection_stats`。frames/detections/score quantileを記録する。
- [x] 🛠 **エラー時対処**: JSON形式が想定と異なる場合は処理を止め、暗黙fallbackしない。

### 手順 17: ByteTrack風の後段trackerを実装し単体テストする
- [x] 🖐 **操作**: `tools/retrack_detections.py` と `tests/test_retrack_detections.py` を追加する。
- [x] 🔎 **確認**: 高score一次マッチ、低score二次マッチ、NMS、IoU+速度予測、summary出力がある。
- [x] 🧪 **テスト**: `uv run --no-sync pytest tests/test_retrack_detections.py -q` と `ruff check` が成功する。
- [x] 🛠 **エラー時対処**: 依存が不足する場合は `uv add` で明示追加する。今回は既存 `scipy` と `opencv-python` で足りたため追加なし。

### 手順 18: 後段trackerパラメータをチューニングする
- [x] 🖐 **操作**: `track_thresh`, `new_track_thresh`, `match_thresh`, `max_age` を複数候補で比較する。
- [x] 🔎 **確認**: raw比で `unique_track_ids / detections` が低下し、長いtrackとbottom-to-top候補が増える。
- [x] 🧪 **テスト**: `manual_retrack_param_grid`。採用paramsとsummary metricsを記録する。
- [x] 🛠 **エラー時対処**: 広いグリッドが重い場合は中断し、初期値周辺と新規track生成抑制方向へ候補を絞る。

### 手順 19: 採用設定で再ID付けJSON/MOT txt/mp4を生成する
- [x] 🖐 **操作**: `just retrack-tracklet-5fps ...` と `just video-retrack-tracklet-5fps 3 0` を実行する。
- [x] 🔎 **確認**: `retrack_bytetrack/tracks.json`, `tracks_mot.txt`, `summary.json`, `tracks.mp4` が生成される。
- [x] 🧪 **テスト**: `manual_retrack_video_metadata`。OpenCVでframes/fps/sizeを確認する。
- [x] 🛠 **エラー時対処**: 動画生成失敗時はJSON/image pathの対応とcodecを確認し、frame dumpは必要時のみ生成する。

### 手順 20: docs/reports更新、品質gate、commit/pushを行う
- [x] 🖐 **操作**: workdocを `reports/` 以下へコピーし、`docs/ONBOARDING.md` に後段trackerの使い方と動画pathを追記する。
- [x] 🔎 **確認**: generated artifactsがgit対象外で、stage対象はsource/test/doc/report/justfileのみ。
- [x] 🧪 **テスト**: 後段tracker関連pytest/ruff、必要に応じて全体pytest、`git diff --check` を実行する。
- [x] 🛠 **エラー時対処**: 生成物がstageされそうなら除外し、`.gitignore` と `git check-ignore` を確認する。

---

## 4. 作業に使用するコマンド参考情報

```bash
date "+%Y-%m-%d %H:%M:%S %Z%z"
cd /home/kasm-user/Desktop/MOTIP
export UV_PROJECT_ENVIRONMENT=$PWD/.venv

uv run --no-sync pytest tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py -q
uv run --no-sync pytest tests/ -q
uv run --no-sync ruff check tools/convert_coco_tracklets_to_pseudomot.py tools/infer_tracklet.py tools/visualize_tracks.py tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py

just build-tracklet-pseudomot-5fps
just loader-tracklet-5fps
just config-tracklet-5fps
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True just train-tracklet-5fps
just infer-tracklet-5fps
just video-tracklet-5fps 3 0
```

---

## 5. 完了の定義（DoD）

- [x] DoD-01: 本作業書が存在し、開始時刻・repo/GPU状態・既存成果物pathが記録されている。
- [x] DoD-02: converterの `--frame-stride` が実装され、既定 `1` の既存挙動とstride変換がテストされている。
- [x] DoD-03: `TomatoTrackletMOT_5fps` datasetが生成され、summaryにsource/selected/skipped/mappingが残る。
- [x] DoD-04: 5FPS configとjust targetがあり、clip内最大unique実測値が `NUM_ID_VOCABULARY` 以下である。
- [x] DoD-05: 60step preflightがfinite loss・OOMなし・EMA/TB/checkpoint正常で通る。
- [x] DoD-06: 5FPS本学習をsaturation判定で停止し、採用checkpoint/log/TB eventが残る。
- [x] DoD-07: 新checkpointで全1219 framesの推論JSON/MOT txtと3fps mp4が生成される。
- [x] DoD-08: 旧新ID proxy metricsを比較し、改善/未改善を数値で記録する。
- [x] DoD-09: pytest/ruffがgreenである。
- [x] DoD-10: generated artifactsがgit管理対象外である。
- [x] DoD-11: commit/push済みで `HEAD == origin/cu118`、working tree cleanである。
- [x] DoD-12: 後段trackerはMOTIP raw `track_id` を使わず、bbox/score/categoryから再ID付けし、単体テストとruffがgreenである。
- [x] DoD-13: 採用設定で後段tracker JSON/MOT txt/summary/mp4が生成され、raw比のID断片化改善とbottom-to-top候補数が数値記録されている。
- [x] DoD-14: `reports/` へworkdocコピー、`docs/ONBOARDING.md` 更新、後段trackerの再現コマンド・成果物pathがgit管理ドキュメントに残る。
- [x] DoD-15: 後段tracker追加分の品質gate・git hygieneが通り、commit/push後に `HEAD == origin/cu118` である。
- [x] DoD-16: Optunaで後段trackerパラメータを探索し、探索結果JSON、best summary、best JSON/MOT txt、3fps mp4が生成されている。
- [x] DoD-17: Optuna best retrack JSONをPseudoMOT datasetへ変換し、frame symlink、`gt.txt`、`seqinfo.ini`、summaryが検証済みである。
- [x] DoD-18: retrack-optuna PseudoMOT用fine-tune config/just targetが追加され、loader smokeと60step preflightがfinite/OOMなしで通る。
- [x] DoD-19: retrack-optuna疑似正解fine-tuneの採用checkpointを決め、saturation判断または完走判断が作業記録に残る。
- [x] DoD-20: 採用checkpointで推論JSON/MOT txt/mp4を生成し、raw/Optuna retrack/fine-tune後のproxy metricsを比較する。
- [x] DoD-21: Optuna/retrack-optuna fine-tune追加分のpytest/ruff/git hygieneがgreenである。
- [x] DoD-22: docs/ONBOARDING.md と `reports/` workdoc copyを更新し、生成物混入なしでcommit/push後に `HEAD == origin/cu118` である。

---

## 6. 作業記録

作業記録上の注意:

- 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` で現在時刻を確認し、正確な日時を記録する。
- 各作業項目の開始時と完了時の両方で記録する。
- 作業内容は具体的なコマンド・操作手順を詳細に記載する。
- 結果・備考欄に成功/失敗、エラー内容、解決方法、重要な気づきを必ず記入する。
- フェーズごとに開始・完了の記録を取る。
- コード変更時は変更ファイル名と変更概要を記録する。
- エラー発生時はエラーメッセージと解決策を詳細に記録する。
- 推測で補完せず、実行したコマンド・観測した出力・編集したファイルに基づいて書く。generated artifactとcommit対象の区別を毎回明示する。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `2026-06-04` | `13:55:53 UTC+0000` | `Codex統括` | 手順1完了: 開始時刻と現状確認 | ✅成功。MOTIPは `/home/kasm-user/Desktop/MOTIP`, branch `cu118`, `## cu118...origin/cu118`, HEAD `e38d990`, GPU `NVIDIA L4, 1MiB/23034MiB, util 0%`。既存成果物: `outputs/tracklet_pseudomot_full/checkpoint_7.pth` 904M, `pretrains/detr_tomato_ckpt7.pth` 156M。 |
| `2026-06-04` | `13:58:18 UTC+0000` | `Codex統括` | DoD-01確認 | ✅成功。本書存在を確認し、DoD-01を `[x]` に更新。MOTIP側には既存未追跡 `workdocs/` があるが、今回の生成物/commit対象ではないため分離して扱う。 |
| `2026-06-04` | `13:59:13 UTC+0000` | `Codex統括` | 手順2完了: stride変換テスト追加とRed確認 | ✅Red確認。`tests/test_tracklet_pseudomot_conversion.py::test_frame_stride_subsamples_frames_and_reindexes_outputs` を追加。`uv run --no-sync pytest tests/test_tracklet_pseudomot_conversion.py -q` は `1 failed, 3 passed`、失敗理由は `TypeError: convert_coco_tracklets_to_pseudomot() got an unexpected keyword argument 'frame_stride'`。 |
| `2026-06-04` | `14:00:08 UTC+0000` | `Codex統括` | 手順3完了: converter `frame_stride` 実装 | ✅成功。`tools/convert_coco_tracklets_to_pseudomot.py` に `frame_stride` / CLI `--frame-stride` を追加。summaryに `source_num_frames`, `frame_rate`, `frame_stride`, `selected_source_image_ids`, `selected_source_file_names`, `source_to_output_frame_ids`, `num_skipped_by_frame_stride` を追加。`uv run --no-sync pytest tests/test_tracklet_pseudomot_conversion.py -q` は `4 passed`。DoD-02を満足。 |
| `2026-06-04` | `14:00:56 UTC+0000` | `Codex統括` | 手順4完了: 5FPS dataset生成just target追加 | ✅成功。`justfile` に `tracklet_5fps_dataset := "datasets/TomatoTrackletMOT_5fps"`、`build-tracklet-pseudomot-5fps`、`loader-tracklet-5fps` を追加。`just --list | rg 'tracklet|5fps'` で両targetを確認。 |
| `2026-06-04` | `14:01:36 UTC+0000` | `Codex統括` | 手順5完了: 5FPS dataset生成・loader検証 | ✅成功。`just build-tracklet-pseudomot-5fps` は `source_num_frames=1219`, `num_frames=204`, `frame_rate=5`, `frame_stride=6`, `num_input_annotations=34101`, `num_skipped_by_frame_stride=28337`, `num_objects=5764`, `num_tracks=1364`, `min/max_frame=1/204` を生成。`seqinfo.ini` は `frameRate=5`, `seqLength=204`。`just loader-tracklet-5fps` は `PseudoMOT.train, 1 sequences, 204 frames.` と `samples 189`。DoD-03を満足。 |
| `2026-06-04` | `14:02:54 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録 | ✅リマインダー表示後の状況記録。手順1-5完了、converter stride実装・5FPS dataset生成済み。未完了は5FPS config/just/preflight/本学習/推論/比較/品質/commit。 |
| `2026-06-04` | `14:02:54 UTC+0000` | `Codex統括` | 手順6完了: 5FPS fine-tune config追加 | ✅成功。`configs/finetune_tracklet_pseudomot_5fps_id16.yaml` を追加。config loadは `PSEUDOMOT_SUB_DIR=TomatoTrackletMOT_5fps`, `SAMPLE_LENGTHS=[16]`, `SAMPLE_INTERVALS=[1]`, `NUM_ID_VOCABULARY=224`, `NUM_TRAINING_IDS=224`, `DETR_PRETRAIN=./pretrains/detr_tomato_ckpt7.pth`, `ID_LOSS_WEIGHT=5.0`, `AUG_NUM_GROUPS=3`, `AMP_DTYPE=bf16`, 当初 `EPOCHS=80`, `SAVE_CHECKPOINT_PER_EPOCH=10`, `EARLY_STOP=False`, `OUTPUTS_DIR=./outputs/tracklet_pseudomot_5fps_id16`, `MAX_TRAIN_STEPS=None`。GT再計測は `max_unique=203`, `truncation_would_fire=False`。 |
| `2026-06-04` | `14:04:38 UTC+0000` | `Codex統括` | 手順7完了: 5FPS学習・推論・動画just target追加 | ✅成功。`justfile` に `tracklet_5fps_config`, `tracklet_5fps_ckpt`, `tracklet_5fps_infer_json` と `config-tracklet-5fps`, `train-tracklet-5fps`, `infer-tracklet-5fps`, `video-tracklet-5fps` を追加。`just config-tracklet-5fps` は `PSEUDOMOT_SUB_DIR=TomatoTrackletMOT_5fps`, `SAMPLE_LENGTHS=[16]`, `SAMPLE_INTERVALS=[1]`, `NUM_ID_VOCABULARY=224`, 当初 `EPOCHS=80`, `AMP_DTYPE=bf16`, `EARLY_STOP=False`, `OUTPUTS_DIR=./outputs/tracklet_pseudomot_5fps_id16` を表示。DoD-04を満足。 |
| `2026-06-04` | `14:10:03 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・手順8完了: 60step preflight | ✅成功。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... train.py --config-path configs/finetune_tracklet_pseudomot_5fps_id16.yaml -u MAX_TRAIN_STEPS=60 EPOCHS=1 OUTPUTS_DIR=./outputs/tracklet_pseudomot_5fps_id16_preflight EXP_NAME=tracklet_pseudomot_5fps_id16_preflight SAVE_CHECKPOINT_PER_EPOCH=1` はexit 0。`PseudoMOT.train, 1 sequences, 204 frames`、DETR pretrain load成功、EMA enabled。最終 `global_step=60`, `loss=28.1794`, `detr_loss=3.0946`, `id_loss=5.0170`, `max_cuda_mem(MB)=14819.5645`。成果物: `checkpoint_0.pth` 925484K, `train/log.txt`, `train/tb/events...`, `config.yaml`。GPUは終了後 `1MiB/23034MiB, util 0%`。DoD-05を満足。 |
| `2026-06-04` | `14:10:03 UTC+0000` | `Codex統括` | 5FPS本学習epoch数調整 | ✅調整。preflight実測は約3.9s/step、189 samples/epochで約12分/epoch。80 epochは10時間を超える見込みのため、ユーザー指定「10時間くらい」に合わせて `EPOCHS=48`, `SCHEDULER_MILESTONES=[30,42]`, `SAVE_CHECKPOINT_PER_EPOCH=8` へ変更。最終checkpointは `checkpoint_47.pth` とし、`justfile` の `tracklet_5fps_ckpt` も同値へ更新。 |
| `2026-06-04` | `14:13:55 UTC+0000` | `Codex統括` | 手順9開始: 5FPS本学習起動 | ✅起動成功。`just train-tracklet-5fps` を実行中（session `48821`）。Runtime configは `EPOCHS=48`, `SAMPLE_LENGTHS=[16]`, `SAMPLE_INTERVALS=[1]`, `NUM_ID_VOCABULARY=224`, `AMP_DTYPE=bf16`, `EARLY_STOP=False`, `SAVE_CHECKPOINT_PER_EPOCH=8`。epoch0 step20時点で `loss=29.4605`, `detr_loss=3.1952`, `id_loss=5.2531`, `max_cuda_mem(MB)=10385.9932`。GPU compute processあり、tracked差分は想定どおり `justfile`, converter, converter test, 5FPS config。 |
| `2026-06-04` | `14:21:44 UTC+0000` | `Codex統括` | 行動カウントリセット後の学習監視 | ✅継続中。14:19:01 UTC+0000に定期リマインダーを表示し行動カウントをリセット。session `48821` はepoch0 step120/189まで進行。直近 `loss=26.3372`, `detr_loss=3.0830`, `id_loss=4.6509`、epoch0平均 `id_loss=4.7628`。`max_cuda_mem(MB)=14819.5645`、GPU process `1223086` が約18.9GB使用。OOM/NaNなし。手順9は未完了のためチェックは据え置き。 |
| `2026-06-04` | `14:58:09 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch0は `id_loss=4.7184`, epoch1は `id_loss=4.5697`, epoch2は `id_loss=4.5604` で正常終了。14:58時点ではepoch3 step100/189、平均 `id_loss=4.5720`。`max_cuda_mem(MB)=14836.5957`、GPU process `1223086` が約18.9GB使用。最初のcheckpoint保存予定はepoch7終了時で、現時点では未生成。OOM/NaNなし。手順9は未完了のためチェックは据え置き。 |
| `2026-06-04` | `16:00:08 UTC+0000` | `Codex統括` | 手順9中間確認: epoch7 checkpoint保存 | ✅継続中。epoch3 `id_loss=4.5508`, epoch4 `4.5326`, epoch5 `4.5313`, epoch6 `4.5335`, epoch7 `4.5301` で正常終了。`outputs/tracklet_pseudomot_5fps_id16/checkpoint_7.pth` 925484Kを確認。出力dirは905M。OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `16:10:50 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch8は `id_loss=4.5381` で正常終了し、epoch9開始。`checkpoint_7.pth` のみ存在（次保存はepoch15終了想定）。`max_cuda_mem(MB)=14836.5957`、OOM/NaNなし。手順9は未完了のためチェックは据え置き。 |
| `2026-06-04` | `17:20:08 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。17:18:39 UTC+0000に定期リマインダーを表示し行動カウントをリセット。epoch9 `id_loss=4.5306`, epoch10 `4.5135`, epoch11 `4.5236`, epoch12 `4.5153`, epoch13 `4.5336` で正常終了。17:20時点ではepoch14 step100/189まで進行し、平均 `id_loss=4.5269`。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14878.7188`、OOM/NaNなし。保存済checkpointは `checkpoint_7.pth` のみで、次保存はepoch15終了想定。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `17:41:31 UTC+0000` | `Codex統括` | 手順9中間確認: epoch15 checkpoint保存 | ✅継続中。epoch14 `id_loss=4.5391`, epoch15 `id_loss=4.5272` で正常終了。`outputs/tracklet_pseudomot_5fps_id16/checkpoint_15.pth` 925488Kを確認し、既存 `checkpoint_7.pth` と合わせて2 checkpoint、出力dirは1.8G。epoch16開始済み。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14878.7188`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `17:43:03 UTC+0000` | `Codex統括` | 手順12準備: ID proxy比較script追加 | ✅準備完了。`tools/compare_track_json.py` を追加し、`justfile` に `compare-tracklet-5fps` と `tracklet_5fps_comparison_json` を追加。旧JSON同士の自己比較で `num_frames=1219`, `num_detections=24119`, `unique_track_ids=24119`, `track_length_mean=1.0` を確認。`uv run --no-sync ruff check tools/compare_track_json.py` は `All checks passed!`。新推論JSON未生成のため手順12/DoD-08は未完了のまま据え置き。 |
| `2026-06-04` | `17:43:35 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch16 step60/189まで進行し、平均 `id_loss=4.5277`。保存済checkpointは `checkpoint_7.pth` と `checkpoint_15.pth`。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14878.7188`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `18:46:06 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch17 `id_loss=4.5364`, epoch18 `4.5322`, epoch19 `4.5286`, epoch20 `4.5038` で正常終了。18:46時点ではepoch21 step20/189まで進行し、平均 `id_loss=4.4857`。保存済checkpointは `checkpoint_7.pth` と `checkpoint_15.pth`、次保存はepoch23終了想定。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14878.7188`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `19:24:07 UTC+0000` | `Codex統括` | 手順9中間確認: epoch23 checkpoint保存 | ✅継続中。epoch21 `id_loss=4.5114`, epoch22 `4.5365`, epoch23 `4.5098` で正常終了。`outputs/tracklet_pseudomot_5fps_id16/checkpoint_23.pth` 904Mを確認し、保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth` の3本。epoch24開始済み。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14878.7188`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `19:40:15 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch24は `id_loss=4.5171` で正常終了し、epoch25開始済み。保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth`。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14879.2178`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `20:57:22 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch25 `id_loss=4.5372`, epoch26 `4.5376`, epoch27 `4.5397`, epoch28 `4.5305`, epoch29 `4.5176`, epoch30 `4.5220` で正常終了。epoch30からscheduler milestone後の低LR帯に入った。保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth`、次保存はepoch31終了想定。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14879.2178`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `21:07:54 UTC+0000` | `Codex統括` | 手順9中間確認: epoch31 checkpoint保存 | ✅継続中。epoch31は `id_loss=4.5263` で正常終了。`outputs/tracklet_pseudomot_5fps_id16/checkpoint_31.pth` 904Mを確認し、保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth`, `checkpoint_31.pth` の4本。epoch32開始済み。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14879.2178`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `22:06:04 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch32 `id_loss=4.5288`, epoch33 `4.5052`, epoch34 `4.5135`, epoch35 `4.5206` で正常終了。22:04時点ではepoch36 step100/189まで進行し、平均 `id_loss=4.5374`。保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth`, `checkpoint_31.pth`、次保存はepoch39終了想定。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14879.2178`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `22:52:47 UTC+0000` | `Codex統括` | 手順9中間確認: epoch39 checkpoint保存 | ✅継続中。epoch36 `id_loss=4.5136`, epoch37 `4.5161`, epoch38 `4.5102`, epoch39 `4.4995` で正常終了。`outputs/tracklet_pseudomot_5fps_id16/checkpoint_39.pth` 904Mを確認し、保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth`, `checkpoint_31.pth`, `checkpoint_39.pth` の5本。epoch40開始済み。`max_cuda_mem(MB)=14879.2178`、OOM/NaNなし。次保存は最終epoch47終了想定。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `23:08:48 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・本学習監視 | ✅継続中。定期リマインダーを表示し行動カウントをリセット。epoch40は `id_loss=4.5214` で正常終了し、23:08時点ではepoch41 step80/189まで進行、平均 `id_loss=4.5301`。保存済checkpointは `checkpoint_7.pth`, `checkpoint_15.pth`, `checkpoint_23.pth`, `checkpoint_31.pth`, `checkpoint_39.pth`。GPU process `1223086` が約18.9GB使用、`max_cuda_mem(MB)=14879.2178`、OOM/NaNなし。手順9は48 epoch完走まで未完了のためチェックは据え置き。 |
| `2026-06-04` | `23:50:55 UTC+0000` | `Codex統括` | 手順9完了: saturation判定による本学習停止・checkpoint採用 | ✅成功。epoch0-43完了ログを集計し、最良はepoch39 `id_loss=4.4995`。直近12 epochは `4.4995-4.5288` の範囲でサチり、直近20 epochの改善は `-0.0084`、直近8 epochの改善は `-0.0119` と悪化。epoch44 step120時点も平均 `id_loss=4.5074` で、追加学習の改善見込みが低いためPID `1223086` をTERM停止。GPU compute processは空になった。採用checkpointは `outputs/tracklet_pseudomot_5fps_id16/checkpoint_39.pth` 904M。`configs/finetune_tracklet_pseudomot_5fps_id16.yaml` は再現用に `EPOCHS=40`, `SCHEDULER_MILESTONES=[30]`、`justfile` の既定checkpointは `checkpoint_39.pth` へ更新。DoD-06を `[x]` に更新。 |
| `2026-06-04` | `23:52:29 UTC+0000` | `Codex統括` | 行動カウントリセット時の状況記録・チェックリスト整合 | ✅成功。定期リマインダーを表示し行動カウントをリセット。手順9のチェックリストをsaturation停止方針に合わせて `[x]` 化し、config確認欄を `EPOCHS=40` に更新。未完了は手順10-15（推論、動画、比較、品質、git hygiene、commit/push）とDoD-07以降。 |
| `2026-06-04` | `23:54:32 UTC+0000` | `Codex統括` | 手順10完了: checkpoint_39で全1219 frame推論 | ✅成功。`just infer-tracklet-5fps` はexit 0。`outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.json` 6,682,284 bytes と `tracks_mot.txt` 1,608,476 bytes / 31,289 rows を生成。JSON確認: frames 1219, detections 31,289, unique_track_ids 31,289, unique_per_detection 1.0。現時点でID一貫性改善は見えておらず、比較工程で正式判定する。 |
| `2026-06-04` | `23:55:33 UTC+0000` | `Codex統括` | 手順11完了: 3fps動画生成・metadata確認 | ✅成功。`just video-tracklet-5fps 3 0` はexit 0。`outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.mp4` を生成。uv環境のOpenCV確認で size 143,900,606 bytes, fps 3.0, frames 1219, width 800, height 600。手順10/11完了によりDoD-07を `[x]` に更新。 |
| `2026-06-04` | `23:56:06 UTC+0000` | `Codex統括` | 手順12完了: 旧新ID proxy比較 | ✅比較完了、改善未達。`just compare-tracklet-5fps` はexit 0、`outputs/tracklet_pseudomot_5fps_id16/infer_30fps/comparison.json` を生成。旧: detections 24,119, unique_track_ids 24,119, unique_ids_per_detection 1.0, track_length_mean 1.0。新: detections 31,289, unique_track_ids 31,289, unique_ids_per_detection 1.0, track_length_mean 1.0。`improved_by_unique_ratio=false`, `improved_by_mean_track_length=false`。5FPS fine-tuneは検出数とscoreは増えたが、ID一貫性proxyは改善しなかった。DoD-08を `[x]` に更新。 |
| `2026-06-04` | `23:56:21 UTC+0000` | `Codex統括` | 依存関係確認: 必要ならuv add | ✅追加不要。ユーザー指示に基づき `pyproject.toml` / `uv.lock` を確認。動画生成・metadata確認に使う `opencv-python>=4.9,<4.11` は既にdependenciesにあり、uv環境で `cv2` import可能。比較script `tools/compare_track_json.py` は標準ライブラリのみ使用。したがって今回の追加作業で新規 `uv add` は不要。 |
| `2026-06-04` | `23:57:41 UTC+0000` | `Codex統括` | 手順13/14完了: 品質gate・git hygiene | ✅成功。周辺テスト `tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py -q` は8 passed。全体 `uv run --no-sync pytest tests/ -q` は43 passed, 8 warnings（既存CLI override warning）。`ruff check tools/convert_coco_tracklets_to_pseudomot.py tools/compare_track_json.py tools/infer_tracklet.py tools/visualize_tracks.py tests/test_tracklet_pseudomot_conversion.py tests/test_pseudo_mot_dataset.py` はAll checks passed。`git diff --check` もexit 0。`git ls-files` でdatasets/outputs/pretrains/checkpoint/mp4/TB eventのtracked生成物なし。`git check-ignore -v` で `datasets/`, `outputs/`, `pretrains/` がignore対象であることを確認。DoD-09/10を `[x]` に更新。 |
| `2026-06-04` | `23:59:10 UTC+0000` | `Codex統括` | 手順15完了: commit & push | ✅成功。stage対象は `configs/finetune_tracklet_pseudomot_5fps_id16.yaml`, `docs/ONBOARDING.md`, `justfile`, `tests/test_tracklet_pseudomot_conversion.py`, `tools/compare_track_json.py`, `tools/convert_coco_tracklets_to_pseudomot.py` の6ファイルのみ。生成物混入grepは空。commit `8012352 Add 5fps tracklet ID fine-tune workflow` を作成し、`git push origin cu118` 成功。`git rev-list --left-right --count HEAD...@{u}` は `0 0`。生成物は `datasets/TomatoTrackletMOT_5fps` 320K, `outputs/tracklet_pseudomot_5fps_id16` 4.6G, preflight 904M がignoredで残存。DoD-11を `[x]` に更新。 |
| `2026-06-05` | `00:24:50 UTC+0000` | `Codex統括` | 追記: mp4動画の所在確認 | ✅確認。今回の5FPS fine-tune評価で見るべき動画は `/home/kasm-user/Desktop/MOTIP/outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.mp4`（`ls -lh`: 138M, Jun 4 23:55）。これは `checkpoint_39.pth` で30FPS全1219 frameへ推論した結果を3fpsで可視化したもの。参考として旧full run動画は `/home/kasm-user/Desktop/MOTIP/outputs/tracklet_pseudomot_full/infer/tracks.mp4`（135M, Jun 4 12:06）で、比較用の旧モデル出力。 |
| `2026-06-05` | `00:33:15 UTC+0000` | `Codex統括` | 追加フェーズ開始: MOTIP検出BBOX列の後段tracker化 | 🔄開始。ユーザー観察どおり `tracks.mp4` はIDが毎フレーム変わるため、MOTIPの `track_id` は使わず `tracks.json` を「frameごとのbbox/score検出列」として扱う。ByteTrack/OC-SORT相当の後段trackerでIDを付け直し、画面下から上まで継続するtrackletが出るかをチューニングする。入力は `/home/kasm-user/Desktop/MOTIP/outputs/tracklet_pseudomot_5fps_id16/infer_30fps/tracks.json`、比較対象動画は同dirの `tracks.mp4`。 |
| `2026-06-05` | `00:37:09 UTC+0000` | `Codex統括` | 手順17完了: 後段tracker実装と単体テスト | ✅成功。`tools/retrack_detections.py` と `tests/test_retrack_detections.py` を追加。MOTIPのraw `track_id` は `source_track_id` として読み捨て、bbox/score/categoryで再ID付けする。単体テストは `2 passed in 0.49s`、`ruff check tools/retrack_detections.py tests/test_retrack_detections.py` はAll checks passed。既存依存 `scipy` を使用し、新規 `uv add` は不要。 |
| `2026-06-05` | `00:37:48 UTC+0000` | `Codex統括` | 手順16/17補足: just target追加と初期設定実行 | ✅成功。`justfile` に `retrack-tracklet-5fps` と `video-retrack-tracklet-5fps` を追加。初期設定 `TRACK=0.75 NEW=0.80 MATCH=0.20 MAX_AGE=30` では raw `31,289 detections / 31,289 unique IDs / ratio 1.0` に対し、後段trackerは `output_detections=29,765`, `unique_track_ids=5,165`, `unique_ids_per_detection=0.173526`, `track_length_max=103`, `long_upward_tracks=48`, `bottom_to_top_tracks=28`。 |
| `2026-06-05` | `00:41:43 UTC+0000` | `Codex統括` | 手順18完了: 後段trackerパラメータ探索 | ✅成功。広いグリッドはCPU時間が長いため停止し、初期値周辺と新規track生成抑制方向へ絞って比較。採用設定は `track_thresh=0.80`, `low_thresh=0.20`, `new_track_thresh=0.95`, `match_thresh=0.10`, `low_match_thresh=0.10`, `max_age=60`, `nms_iou=0.70`, `velocity_weight=1.0`。この設定で `unique_ids_per_detection=0.062644`, `track_length_mean=15.963226`, `median=11.5`, `max=114`, `num_tracks_ge_30=149`, `num_tracks_ge_60=47`, `long_upward_tracks=130`, `bottom_to_top_tracks=101`。 |
| `2026-06-05` | `00:42:27 UTC+0000` | `Codex統括` | 手順19完了: 再ID付けJSON/MOT txt/mp4生成 | ✅成功。`just retrack-tracklet-5fps 0.80 0.20 0.95 0.10 0.10 60 0.70 1.0` で `outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/{tracks.json,tracks_mot.txt,summary.json}` を生成。`just video-retrack-tracklet-5fps 3 0` で `outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/tracks.mp4` を生成。OpenCV確認: opened=True, frames=1219, fps=3.0, width=800, height=600。ファイルサイズは mp4 127M, JSON 5.1M, MOT txt 1.2M。DoD-12/13を満足。 |
| `2026-06-05` | `00:45:07 UTC+0000` | `Codex統括` | 手順20一部完了: docs/reports更新 | ✅成功。`docs/ONBOARDING.md` に `just retrack-tracklet-5fps ...` と `just video-retrack-tracklet-5fps 3 0`、採用params、成果物path、raw比改善metrics、動画pathを追記。`reports/workdoc_Jun04-2026_motip_5fps_id_finetune.md` へ本書をコピー。DoD-14を満足。残りは品質gate、git hygiene、commit/push。 |
| `2026-06-05` | `00:47:10 UTC+0000` | `Codex統括` | 手順18/19補足: tuned設定をjust既定値へ反映 | ✅成功。`retrack-tracklet-5fps` の既定値を採用設定 `TRACK=0.80 LOW=0.20 NEW=0.95 MATCH=0.10 LOW_MATCH=0.10 MAX_AGE=60 NMS=0.70 VEL=1.0` へ変更し、引数なし `just retrack-tracklet-5fps` で同じsummaryを再生成できることを確認。代表bottom-to-top候補では例としてtrack `1308` が frame `982 -> 1219`, y-center `524.8 -> 28.8`、track `1` が frame `1 -> 102`, y-center `528.3 -> 9.2`。 |
| `2026-06-05` | `00:47:43 UTC+0000` | `Codex統括` | 手順20品質gate完了 | ✅成功。`pytest tests/test_retrack_detections.py -q` は `2 passed in 0.49s`、`ruff check tools/retrack_detections.py tests/test_retrack_detections.py` はAll checks passed、`git diff --check` はexit 0。全体pytestは直前に `45 passed, 8 warnings`。`git ls-files | rg '^(outputs|datasets|pretrains)/|\\.pth$|\\.mp4$|events\\.out|^\\.venv/'` は空で、retrack成果物は `.gitignore:28:outputs/` 対象。残りはcommit/pushのみ。 |
| `2026-06-05` | `00:48:35 UTC+0000` | `Codex統括` | 手順20完了予定: commit/push gate | ✅stage対象をsource/test/doc/report/justfileに限定してcommit/pushする。生成物は `outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/` 配下に残すが `.gitignore` 対象でgit外。commit/push後に `git rev-list --left-right --count HEAD...@{u}` が `0 0` であることを最終確認する。 |
| `2026-06-05` | `01:24:05 UTC+0000` | `Codex統括` | Optuna後段tracker探索とretrack疑似正解fine-tuneの中間記録 | 🔄継続中。`uv add optuna` により `pyproject.toml`/`uv.lock` を更新。`tools/tune_retrack_detections.py` と `tests/test_tune_retrack_detections.py` を追加し、80 trials/240sでOptuna探索を実行。best paramsは `track_thresh=0.85`, `low_thresh=0.4`, `new_track_thresh=0.85`, `match_thresh=0.01`, `low_match_thresh=0.01`, `max_age=240`, `nms_iou=0.55`, `velocity_weight=0.0`, `velocity_momentum=0.65`。best summaryは input 31,289 / output 30,379 / unique 827 / mean length 36.733978 / median 32.0 / max 151 / tracks>=120 18 / long_upward 196 / bottom_to_top 67。成果物は `outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/{tracks.json,tracks_mot.txt,summary.json,optuna_study.json,tracks.mp4}`。DoD-16を満足。 |
| `2026-06-05` | `01:24:05 UTC+0000` | `Codex統括` | Optuna best retrack JSONのPseudoMOT化 | ✅成功。`tools/convert_track_json_to_pseudomot.py` と `tests/test_track_json_pseudomot_conversion.py` を追加。初回preflightで相対symlinkが壊れる問題を発見し、source image pathを `resolve()` した絶対symlinkへ修正。`just build-tracklet-pseudomot-retrack-optuna` により `datasets/TomatoTrackletMOT_retrack_optuna` を生成。summaryは sequence `nyx660_jun04_retrack_optuna`, frames 1219, objects 30379, tracks 827, empty frames 0。`find ... -xtype l` でbroken symlink 0。DoD-17を満足。 |
| `2026-06-05` | `01:24:05 UTC+0000` | `Codex統括` | retrack-optuna fine-tune config/preflight | ✅成功。Optuna疑似正解ではclip内uniqueが増え、計測値は SL=8/interval1 max_unique 210、SL=10/interval2 max_unique 296、SL=16/interval1 max_unique 327。既存checkpoint `checkpoint_39.pth` のID vocab 224へstrict resumeするため、`configs/finetune_tracklet_pseudomot_retrack_optuna.yaml` は `SAMPLE_LENGTHS=[8]`, `SAMPLE_INTERVALS=[1]`, `NUM_ID_VOCABULARY=224`, `NUM_TRAINING_IDS=224`, `PSEUDOMOT_SUB_DIR=TomatoTrackletMOT_retrack_optuna`, `RESUME=./outputs/tracklet_pseudomot_5fps_id16/checkpoint_39.pth`, optimizer/scheduler resume falseで追加。`just loader-tracklet-retrack-optuna` は `PseudoMOT.train, 1 sequences, 1219 frames.`, samples 1212。60step preflightは `global_step 7560 -> 7620`, `id_loss` 約4.00、`max_cuda_mem(MB)=4635.9404`, OOMなし。DoD-18を満足。 |
| `2026-06-05` | `01:24:05 UTC+0000` | `Codex統括` | retrack-optuna疑似正解fine-tune本学習監視 | 🔄継続中。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... train.py --config-path configs/finetune_tracklet_pseudomot_retrack_optuna.yaml` をsession `72798` で実行中。`checkpoint_39.pth` から `start_epoch=40`, `global_step=7560` で再開。01:24時点ではepoch40 step480/1212まで進行し、平均 `id_loss` は4.18付近、`detr_loss` は2.76付近、`max_cuda_mem(MB)=4635.9404`。preflight時の `id_loss` 約4.00より悪化気味でサチり懸念があるため、少なくともepoch40 checkpoint生成後に採用/停止判断を行う。DoD-19以降は未完了。 |
| `2026-06-05` | `01:32:56 UTC+0000` | `Codex統括` | 行動カウントリセット時のretrack-optuna fine-tune監視 | 🔄継続中。定期リマインダーを表示し行動カウントをリセット。epoch40 step940/1212まで進行。平均 `id_loss=4.1845`, `detr_loss=2.7640`, `loss=23.6864`, `max_cuda_mem(MB)=4826.7515`。GPU process `1789915` が約6.2GB使用中。`checkpoint_40.pth` は未生成。preflight `id_loss` 約4.00より高く、追加epochを続ける価値は低い見込み。epoch40保存後に停止し、採用checkpointとして評価する予定。 |
| `2026-06-05` | `01:40:09 UTC+0000` | `Codex統括` | retrack-optuna fine-tune採用checkpoint確定・推論開始 | ✅DoD-19完了。epoch40は `loss=23.7169`, `detr_loss=2.7532`, `id_loss=4.1927`, `max_cuda_mem(MB)=4831.9561`, 所要 `0:23:13` で正常終了し、`outputs/tracklet_pseudomot_retrack_optuna_ft/checkpoint_40.pth` 904Mを生成。直後epoch41へ入ったためPID `1789915` をTERMし、終了コード143で停止、GPU compute processは空。preflightの `id_loss` 約4.00より高く、epoch40内でも4.18-4.19で横ばいだったため、追加epochはサチり/悪化リスクが高いと判断し `checkpoint_40.pth` を採用。`configs/finetune_tracklet_pseudomot_retrack_optuna.yaml` は再現用に `EPOCHS=41` へ修正。`just infer-tracklet-retrack-optuna-ft 0 fp32` を開始し、DoD-20評価へ移行中。 |
| `2026-06-05` | `01:40:09 UTC+0000` | `Codex統括` | retrack-optuna fine-tune後の推論・動画・比較 | ✅DoD-20完了。`just infer-tracklet-retrack-optuna-ft 0 fp32` はexit 0で、`outputs/tracklet_pseudomot_retrack_optuna_ft/infer_30fps/tracks.json` と `tracks_mot.txt` を生成。結果は frames 1219, detections 25,712, unique IDs 25,712, `track_length_mean=1.0` で、MOTIP raw IDは依然として毎検出新規。`just video-tracklet-retrack-optuna-ft 3 0` は `tracks.mp4` 142,911,908 bytes, 1219 frames, 3fps, 800x600を生成。`just compare-tracklet-retrack-optuna-ft` によるOptuna疑似正解との比較は、Optuna側: detections 30,379, unique IDs 827, `unique_ids_per_detection=0.027223`, `track_length_mean=36.733978`, max 151, tracks>=60 104, tracks>=120 18, long_upward 196, bottom_to_top 67。fine-tune後raw側: detections 25,712, unique IDs 25,712, `unique_ids_per_detection=1.0`, `track_length_mean=1.0`。結論: 後段tracker自体は長trackletを作るが、それを1epochの疑似正解fine-tuneでMOTIPの内部ID headへ転写することはできなかった。実用出力としてはOptuna retrack JSON/mp4を採用し、MOTIP側は追加設計（ID loss/decoder/teacher forcing/label noise対策）が必要。 |
| `2026-06-05` | `01:40:09 UTC+0000` | `Codex統括` | Optuna/retrack-optuna追加分の品質gate | ✅DoD-21完了。関連テスト `pytest tests/test_retrack_detections.py tests/test_tune_retrack_detections.py tests/test_track_json_pseudomot_conversion.py -q` は `6 passed in 0.56s`。`ruff check tools/retrack_detections.py tools/tune_retrack_detections.py tools/convert_track_json_to_pseudomot.py tests/test_retrack_detections.py tests/test_tune_retrack_detections.py tests/test_track_json_pseudomot_conversion.py` はAll checks passed。`git diff --check` はexit 0。生成物混入確認 `git ls-files | rg '^(outputs|datasets|pretrains)/|\\.pth$|\\.mp4$|events\\.out|^\\.venv/'` は空。全体 `pytest tests/ -q` は `49 passed, 8 warnings`（既存CLI override warning）。 |
| `2026-06-05` | `01:44:42 UTC+0000` | `Codex統括` | Optuna/retrack-optuna追加分のcommit & push | ✅DoD-22完了。stage対象は `configs/finetune_tracklet_pseudomot_retrack_optuna.yaml`, `docs/ONBOARDING.md`, `justfile`, `pyproject.toml`, `uv.lock`, `reports/workdoc_Jun04-2026_motip_5fps_id_finetune.md`, `tests/test_track_json_pseudomot_conversion.py`, `tests/test_tune_retrack_detections.py`, `tools/convert_track_json_to_pseudomot.py`, `tools/tune_retrack_detections.py` の10ファイルのみ。staged生成物grepは空。commit `64f3c50 Add optuna retracking pseudo-label workflow` を作成し、`git push origin cu118` 成功。`git rev-list --left-right --count HEAD...@{u}` は `0 0`、`git status --short --branch` は `## cu118...origin/cu118`。生成物は `outputs/`, `datasets/`, checkpoint/mp4としてgit外に残存。 |

## 7. サブエージェントロスター

| agent_id | name | scope | workspace | branch_or_context | allowed_actions | forbidden_actions | status | last_update | evidence_returned |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| local | Codex統括 | 実装・検証・記録・commit/push | `/home/kasm-user/Desktop/MOTIP` | `cu118` | source/config/test/doc/justfile編集、uv/just実行、生成物作成、commit/push | 生成物のgit追加、旧成果物削除、暗黙fallback | active | `2026-06-04 13:55:53 UTC+0000` | 本書 |
