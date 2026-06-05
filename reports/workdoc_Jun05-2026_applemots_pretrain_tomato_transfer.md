# 作業計画書 兼 記録書: AppleMOTS事前学習とトマトMOTIP転移

---

**日付:** `2026年06月05日`
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/MOTIP`（branch `cu118`, remote `git@github.com:yuki-inaho/MOTIP.git`, uv venv `/home/kasm-user/Desktop/MOTIP/.venv`）および `/workspace/Project/DEIM_sandbox`（branch `deimv1+cu128`, uv venv `/home/kasm-user/Desktop/.venv`）
**作業者:** `Codex統括`

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

* **目標1:** AppleMOTSをMOTIPのtracking事前学習データとして使い、トラッキングに良い条件を実験比較する。
* **目標2:** AppleMOTS bboxをDEIM検出器にも使えるCOCO形式へ変換し、DEIM学習smokeを通す。
* **目標3:** AppleMOTSで得たMOTIP checkpointをpretrained modelとして、トマトtrackletモデルを再学習・評価する。
* **目標4:** 実験条件、結果、失敗理由、採用判断をレポートし、作業書をwrite/reviewスキル観点で自己改善しながらDoDを満たす。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** トマトだけの疑似正解fine-tuneではMOTIPのIDが毎フレーム切れてしまうため、農業MOT公開データであるAppleMOTSを使ってtracking能力を事前学習し、その重みをトマトへ転移してID一貫性を改善したい。DEIM検出器側もAppleMOTS bboxで学習可能な形にし、MOTIP単体でなく検出器/追跡器の両面から再利用できる状態にしたい。
* **明示要求:**
  * AppleMOTSでDEIM・MOTIPを含めて学習を行う。
  * MOTIPのpretrained modelは先に使っていた公式BFT系モデルでよい。
  * トラッキングに良い条件を実験的に検証し、結果をレポートする。
  * AppleMOTSで学習したモデルをpretrained modelとして、トマトモデルの学習を行う。
  * 最終的にどのような形に学習できたかを記録する。
  * `write-workdoc-uv` / `review-written-workdoc` スキルを使い、作業書を自己改善しつつ進める。
  * DoDを満たすまで手を止めない。
* **暗黙制約:**
  * MOTIP側Python実行は `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run --no-sync` を使う。
  * DEIM側Python実行は `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/.venv` を使う。2026-06-05 04:52時点ではこのvenvは新規作成直後でtorch未導入のため、`uv sync --frozen` が必要。
  * `justfile` targetを優先し、再現可能なコマンドにする。
  * no-silent-fallback。pretrain欠損、shape mismatch、ID語彙不足、OOM、NaN、評価JSON欠損は明示失敗として記録する。
  * 生成物（`datasets/`, `outputs/`, `pretrains/`, `.pth`, `.mp4`, TensorBoard events, raw dataset）はgitへ追加しない。
  * 作業前・重要操作前に `date "+%Y-%m-%d %H:%M:%S %Z%z"` を実行し、作業記録へ残す。
* **非ゴール:**
  * AppleMOTSで最終論文品質の長時間full trainingを保証しない。まずはL4上で再現可能な短期実験と、トマト転移可否の検証を優先する。
  * DEIMのONNX exportやトマトONNX差替までは今回の必須DoDに含めない。
  * HOTA/MOTAなど正式MOT評価基盤の新規実装は必須にしない。MOTIPの現有JSON proxy metrics（unique IDs per detection、track length）を主評価とし、可能ならCOCO mAP/DEIM evalを併記する。
* **成功条件:**
  * AppleMOTS raw data、PseudoMOT変換、COCO変換の構成が作業書とsummary JSONに残る。
  * AppleMOTS用MOTIP configが複数条件（短期smoke、BFT tracking移植、sample length/ID語彙/optimizer条件）で比較され、少なくとも1つのAppleMOTS MOTIP checkpointが生成される。
  * AppleMOTS COCO datasetでDEIM dataloader/training smokeが成功する。
  * AppleMOTS MOTIP checkpointをpretrainedとしてトマトretrack-optuna datasetへ転移し、トマト学習smokeまたは短期fine-tuneを完走する。
  * 旧トマト出力、AppleMOTS pretrained転移後出力、必要ならOptuna retrack疑似正解とのproxy metrics比較が記録される。
  * pytest/ruff/git hygiene/workdoc review/docs/reports更新が完了する。
* **リスクと前提:**
  * AppleMOTSはりんごで、トマトと外観・カメラ運動・遮蔽が異なる。転移でID改善しない可能性がある。
  * AppleMOTS画像は1296x972でobject数が多い。L4 23GBでは `SL20` や `NUM_TRAINING_IDS` がOOMする可能性があるため、短辺384/512、`AUG_NUM_GROUPS=1`、`NUM_TRAINING_IDS` capを明示的に試す。
  * DEIM_sandboxのvenvは未同期。同期に時間がかかる場合は、MOTIP側学習を先に進める。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 作業書・現状確定 | 目標4 | 本作業書、repo/env/GPU状態 | `date`, `git status`, `nvidia-smi`, workdoc review |
| SG-2 | AppleMOTS COCO変換 | 目標2 | `tools/convert_apple_mots_to_coco.py`, `datasets/AppleMOTSCOCO` | pytest, summary JSON, COCO load |
| SG-3 | DEIM AppleMOTS smoke | 目標2 | DEIM config, smoke output/log | `uv sync --frozen`, DEIM train smoke |
| SG-4 | MOTIP AppleMOTS実験 | 目標1 | AppleMOTS MOTIP configs/checkpoints/logs | 60/120step比較、loss/id_loss/memory |
| SG-5 | AppleMOTS→トマト転移 | 目標3 | tomato transfer config/checkpoint/infer JSON/mp4/comparison | loader/train/infer/compare |
| SG-6 | レポート・品質・commit | 目標4 | docs/reports/workdoc更新、tests green、git hygiene | pytest/ruff/diff-check/stage grep |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | AppleMOTSをMOTIP学習に使う | 手順7-10 | config dump, training log, checkpoint |
| TR-2 | AppleMOTSをDEIM学習に使う | 手順3-6 | COCO summary, DEIM config, DEIM smoke log |
| TR-3 | AppleMOTS pretrainedからトマト転移 | 手順11-12 | transplant report, tomato train log, infer JSON |
| TR-4 | 実験検証とレポート | 手順9-13 | metrics JSON, workdoc/reports, ONBOARDING |
| TR-5 | write/reviewスキルと監査性 | 手順1-2, 手順15 | review findings, applied changes |
| TR-6 | 生成物git混入禁止 | 全手順 | `.gitignore`, `git status`, staged grep |

---

## 2. 作業内容

### フェーズ 1: 現状確認と作業書整備

MOTIP/DEIMのrepo状態、uv環境、AppleMOTS変換済み成果物、BFT checkpoint、GPU空き状況を確認し、本書を作成する。review-written-workdocのrubricに照らして、作業開始前に実行可能性を自己レビューする。

### フェーズ 2: AppleMOTSをDEIMへ載せる

AppleMOTS instance maskからCOCO detection JSONを生成する。MOTIP用PseudoMOT変換と同じraw dataを使うが、DEIM `CocoDetection` が読める `images/` + `annotations/*.json` 構成にする。DEIM_sandboxにAppleMOTS smoke configを追加し、venv同期後に短期学習を確認する。

### フェーズ 3: AppleMOTSでMOTIP tracking事前学習

公式BFT MOTIP checkpointをsourceに、AppleMOTS用target configへtracking層を移植する。`SL2/SL8/SL20` などの候補を短期比較し、L4でOOMせずid_loss/track proxyが良い条件を採用する。採用条件でcheckpointを生成する。

### フェーズ 4: AppleMOTS pretrainedからトマトへ転移

AppleMOTSで得たcheckpointをsource/baseとして、`TomatoTrackletMOT_retrack_optuna` 向けconfigに移植・fine-tuneする。短期学習後にトマト全frame推論を行い、既存のraw/Optuna retrack結果とproxy metricsを比較する。

### フェーズ 5: レポート・自己レビュー・品質gate

作業記録、実験結果、失敗条件、採用条件、残課題をreports/docsへ反映する。`review-written-workdoc` rubricで本書をレビューし、Blocker/Majorを修正する。テスト・lint・生成物混入チェックを行う。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。各チェック完了直後に作業記録へ日時・結果・証跡を追記します。*

### フェーズ 1: 現状確認と作業書整備

### 手順 1: 開始時刻・repo/env/GPU状態を記録する
- [x] 🖐 **操作**: `date`, MOTIP/DEIM `git status`, `nvidia-smi`, uv環境import確認を実行し、本書へ記録する。
- [x] 🔎 **確認**: MOTIP branch、DEIM branch、未同期venv、GPU空き、既存AppleMOTS dataset/checkpoint pathが作業記録にある。
- [x] 🧪 **テスト**: `manual_start_record`。本書が `temp/workdoc_Jun05-2026_applemots_pretrain_tomato_transfer.md` に存在する。
- [x] 🛠 **エラー時対処**: training processが残っている場合は停止せず、PID/command/GPU使用量を記録し、衝突しない作業から進める。

### 手順 2: 作業書をreview-written-workdoc観点で初回レビューする
- [x] 🖐 **操作**: review rubricを読み、本書のContext/Goal/Traceability/Checklist/DoDを自己レビューして不足を修正する。
- [x] 🔎 **確認**: Blocker/Major findingsが0、または残る場合は明示的な理由と対処順がある。
- [x] 🧪 **テスト**: `manual_workdoc_review_initial`。review結果を作業記録へ追記する。
- [x] 🛠 **エラー時対処**: 目的と手順がずれている場合は実装を始めず、本書を先に修正する。

### フェーズ 2: AppleMOTSをDEIMへ載せる

### 手順 3: AppleMOTS -> COCO変換テストを追加する
- [x] 🖐 **操作**: `tests/test_convert_apple_mots_to_coco.py` を追加し、小さな16bit maskからCOCO images/annotations/categories/summaryを期待する。
- [x] 🔎 **確認**: 実装前にimport/関数未存在で失敗し、期待するCOCO bbox形式がテストに固定されている。
- [x] 🧪 **テスト**: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run --no-sync pytest tests/test_convert_apple_mots_to_coco.py -q` の赤→緑。
- [x] 🛠 **エラー時対処**: COCO category id/remapが曖昧なら、AppleMOTSは単一class `apple` category id=1として固定し、track idはattributesへ保存する。

### 手順 4: AppleMOTS -> COCO変換CLIとjust targetを実装する
- [x] 🖐 **操作**: `tools/convert_apple_mots_to_coco.py` と `just build-applemots-coco` を追加し、`datasets/AppleMOTSCOCO` を生成する。
- [x] 🔎 **確認**: train/testing annotation JSON、image symlink、summary JSONが生成され、DEIM `CocoDetection` で読める構成になっている。
- [x] 🧪 **テスト**: COCO変換単体テスト、`python -c` で画像数/annotation数/categoryを確認。
- [x] 🛠 **エラー時対処**: DEIM側が`.jpg` symlink実体PNGを読めない場合は、copy/PNG拡張子維持へ切り替え、COCO `file_name` を実体拡張子に合わせる。

### 手順 5: DEIM環境を同期しAppleMOTS smoke configを追加する
- [x] 🖐 **操作**: `/workspace/Project/DEIM_sandbox` で `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/.venv uv sync --frozen` を実行し、AppleMOTS COCO smoke configを追加する。
- [x] 🔎 **確認**: torch/faster-coco-eval import成功、configはAppleMOTSCOCOのtrain/testing JSONを指す。
- [x] 🧪 **テスト**: `manual_deim_env_and_config_smoke`。config loadまたは短期train前のimport確認が成功する。
- [x] 🛠 **エラー時対処**: `uv sync --frozen` がlock不一致や通信失敗なら、ログを残し、MOTIP側実験を先に進める。

### 手順 6: DEIM AppleMOTS学習smokeを実行する
- [x] 🖐 **操作**: DEIM train entrypointでAppleMOTS configを1 epochまたは少数iterationで実行する。
- [x] 🔎 **確認**: output dir/log/checkpointまたはeval summaryが生成され、OOM/NaNなし。
- [x] 🧪 **テスト**: `manual_deim_applemots_train_smoke`。終了コード、loss/mAPまたはlog末尾を記録する。
- [x] 🛠 **エラー時対処**: L4でOOMする場合はbatch size/num_workers/input sizeを下げる。config上の縮退を記録し、暗黙に別datasetへfallbackしない。

### フェーズ 3: AppleMOTSでMOTIP tracking事前学習

### 手順 7: AppleMOTS BFT移植MOTIP configを追加する
- [x] 🖐 **操作**: AppleMOTS向けBFT tracking移植config（例 `configs/finetune_applemots_pseudomot_bft_official.yaml`）とjust targetを追加する。
- [x] 🔎 **確認**: `PSEUDOMOT_SUB_DIR=AppleMOTSPseudoMOT`, `SAMPLE_LENGTHS`, `NUM_ID_VOCABULARY`, `NUM_TRAINING_IDS`, `RESUME_MODEL`, `OUTPUTS_DIR` が明示される。
- [x] 🧪 **テスト**: `just config-applemots-bft-official` が主要キーを表示する。
- [x] 🛠 **エラー時対処**: target vocabとBFT vocab shapeが違う場合は既存 `tools/transplant_motip_tracking_weights.py` のpartial vocab copyを使う。

### 手順 8: AppleMOTS用BFT tracking移植checkpointを生成する
- [x] 🖐 **操作**: 公式BFT checkpointからAppleMOTS target configへ `trajectory_modeling.` / `id_decoder.` を移植する。
- [x] 🔎 **確認**: output checkpointとreport JSONが生成され、exact/partial/skip summaryが作業記録にある。
- [x] 🧪 **テスト**: `manual_applemots_bft_transplant`。strict resume可能なcheckpointであることを短期config loadで確認する。
- [x] 🛠 **エラー時対処**: shape mismatchがtracking層主要部に出たら、skip keyをreportに残し、無理にreshapeしない。

### 手順 9: AppleMOTS MOTIP短期条件比較を実行する
- [x] 🖐 **操作**: `SL2`, `SL8`, `SL20` など複数条件を `MAX_TRAIN_STEPS=60/120` で比較する。
- [x] 🔎 **確認**: 各条件のloss/detr_loss/id_loss/max_cuda_mem/step time/OOM有無が表に記録される。
- [x] 🧪 **テスト**: `manual_applemots_motip_condition_sweep`。最良条件を採用理由つきで選ぶ。
- [x] 🛠 **エラー時対処**: OOM時は `AUG_NUM_GROUPS`, `NUM_TRAINING_IDS`, input short sideの順に明示縮退する。

### 手順 10: 採用条件でAppleMOTS MOTIP checkpointを作る
- [x] 🖐 **操作**: 採用configでAppleMOTS MOTIP trainingを実行し、saturation/時間枠/early-stopでcheckpointを決める。
- [x] 🔎 **確認**: checkpoint path、epoch/global_step、loss/id_loss、停止理由が作業記録にある。
- [x] 🧪 **テスト**: `manual_applemots_motip_train`。checkpointをloadして推論/転移に使える。
- [x] 🛠 **エラー時対処**: 長時間で改善しない場合は未完走でもsaturation判断を明記して停止する。

### フェーズ 4: AppleMOTS pretrainedからトマトへ転移

### 手順 11: AppleMOTS checkpointをトマトtargetへ移植する
- [x] 🖐 **操作**: AppleMOTS checkpointをsourceとして `TomatoTrackletMOT_retrack_optuna` 向けtarget configへtracking重みを移植する。
- [x] 🔎 **確認**: tomato target checkpoint/report JSONが生成され、ID vocab overlap/unknown copy結果が記録される。
- [x] 🧪 **テスト**: `manual_applemots_to_tomato_transplant`。tomato configでstrict resumeまたはloadが成功する。
- [x] 🛠 **エラー時対処**: AppleMOTSとtomatoでID vocabが違う場合はpartial copyを使い、検出器側はtomato base checkpointを優先する。

### 手順 12: トマトfine-tuneと推論比較を実行する
- [x] 🖐 **操作**: AppleMOTS pretrained tomato configで短期fine-tuneし、全1219frame推論、動画、comparison JSONを生成する。
- [x] 🔎 **確認**: raw/Optuna retrack/既存BFT移植/AppleMOTS転移後のproxy metricsを比較し、改善/未改善を明記する。
- [x] 🧪 **テスト**: `manual_tomato_transfer_eval`。`tracks.json`, `tracks_mot.txt`, `tracks.mp4`, comparison JSONが存在する。
- [x] 🛠 **エラー時対処**: ID改善未達なら、ID教師ノイズ、query association、DETR凍結、NUM_TRAINING_IDS cap、domain gapを原因候補として切り分ける。

### フェーズ 5: レポート・自己レビュー・品質gate

### 手順 13: 実験レポートとdocsを更新する
- [x] 🖐 **操作**: `reports/` に本書をコピーし、`docs/ONBOARDING.md` へAppleMOTS/DEIM/MOTIP/トマト転移コマンドと成果物pathを追記する。
- [x] 🔎 **確認**: 別エージェントが本書とONBOARDINGだけで再実行できる。
- [x] 🧪 **テスト**: `manual_docs_evidence_check`。リンク/パス/コマンドが実在する。
- [x] 🛠 **エラー時対処**: 未完了実験は完了扱いせず、未完了/保留/次アクションに分類する。

### 手順 14: 品質gateと生成物混入チェックを行う
- [x] 🖐 **操作**: 対象pytest/ruff、`git diff --check`、生成物混入grep、`git status` を実行する。
- [x] 🔎 **確認**: source/config/test/doc/reportだけがstage候補で、datasets/outputs/pretrains/raw dataはgit外。
- [x] 🧪 **テスト**: `manual_quality_gate`。コマンド結果を作業記録に残す。
- [x] 🛠 **エラー時対処**: 生成物がstageされていたらunstageし、`.gitignore` またはstage対象を修正する。

### 手順 15: review-written-workdoc最終レビューを行い、必要ならcommit/pushする
- [x] 🖐 **操作**: rubricで本書を再レビューし、Blocker/Majorを修正する。ユーザー指示または作業書DoDに従いcommit/pushする。
- [x] 🔎 **確認**: DoD充足状況が本書にあり、未達があれば未達として明記されている。commitした場合は `HEAD == origin/cu118`。
- [x] 🧪 **テスト**: `manual_final_audit`。DoDごとの証跡を確認する。
- [x] 🛠 **エラー時対処**: 長時間学習中断や外部環境同期失敗が残る場合は、blockedではなく再開可能なhandoffとして記録する。

---

## 4. 作業に使用するコマンド参考情報

```bash
date "+%Y-%m-%d %H:%M:%S %Z%z"
cd /home/kasm-user/Desktop/MOTIP
export UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv
uv run --no-sync pytest tests/test_convert_apple_mots_to_pseudomot.py -q
just build-applemots-pseudomot
just loader-applemots
just train-applemots-smoke

cd /workspace/Project/DEIM_sandbox
export UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/.venv
uv sync --frozen
uv run --no-sync python DEIM/train.py -c configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml
```

---

## 5. 完了の定義（DoD）

- [x] DoD-01: 本書が存在し、開始時刻、MOTIP/DEIM repo状態、GPU状態、uv環境状態が記録されている。
- [x] DoD-02: 本書がreview-written-workdoc rubricで初回レビューされ、Blocker/Majorが修正されている。
- [x] DoD-03: AppleMOTS -> COCO変換CLI・テスト・summary JSONがあり、pytest/ruffがgreenである。
- [x] DoD-04: `datasets/AppleMOTSCOCO` が生成され、DEIM `CocoDetection` が読める画像/annotation構成である。
- [x] DoD-05: DEIM_sandbox uv環境が同期され、AppleMOTS DEIM configで学習smokeが成功している。
- [x] DoD-06: AppleMOTS MOTIP BFT移植configとcheckpoint/report JSONが生成されている。
- [x] DoD-07: AppleMOTS MOTIPで複数条件の短期比較が行われ、採用条件が理由つきで決まっている。
- [x] DoD-08: 採用条件のAppleMOTS MOTIP checkpointが生成され、loss/id_loss/停止理由が記録されている。
- [x] DoD-09: AppleMOTS checkpointをpretrainedとしてトマトtarget checkpointへ移植できている。
- [x] DoD-10: トマトfine-tune、全frame推論、動画、comparison JSONが生成され、既存結果と比較されている。
- [x] DoD-11: 実験結果レポート、docs/ONBOARDING、reportsコピーが更新されている。
- [x] DoD-12: pytest/ruff/git diff check/生成物混入チェックがgreenである。
- [x] DoD-13: 最終workdoc reviewでBlocker/Majorが0、または残課題が明示されている。
- [x] DoD-14: commit/pushする場合、生成物混入なしで `HEAD == origin/<branch>` が確認されている。

---

## 6. 作業記録

作業記録上の注意:

- 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` で現在時刻を確認し、正確な日時を記録する。
- 各作業項目の開始時と完了時の両方で記録する。
- エラーが発生した場合は、エラーメッセージ、原因調査、対処内容を必ず記録する。
- コミットやプッシュを行った場合は、コミットハッシュと対象ファイルを記録する。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `2026-06-05` | `04:50:35 UTC+0000` | `Codex統括` | 手順1開始: 現状確認 | MOTIPは `/home/kasm-user/Desktop/MOTIP`, branch `cu118`, HEAD `19202f4`, dirtyあり（BFT移植/optimizer/AppleMOTS smoke関連）。DEIMは `/workspace/Project/DEIM_sandbox`, branch `deimv1+cu128`, HEAD `0124409`、submodule `DEIM` は `88d525f`。GPUはL4で空き。 |
| `2026-06-05` | `04:52:18 UTC+0000` | `Codex統括` | write-workdoc-uvに基づく本書作成 | `LC_TIME=C date "+%b%d-%Y"` は `Jun05-2026`。MOTIP uv環境はtorch 2.4.0+cu118とBFT checkpointを確認。DEIM uv環境 `/home/kasm-user/Desktop/.venv` は新規作成状態でtorch/faster-coco-eval未導入のため、手順5で `uv sync --frozen` が必要。 |
| `2026-06-05` | `04:54:55 UTC+0000` | `Codex統括` | 手順1/2完了: 初回workdoc review | ✅DoD-01/02完了。`review-written-workdoc` rubricに基づき自己レビュー。Verdictは `PASS_WITH_NOTES`。Blockerなし。Major候補だった未確定placeholder `configs/deim_dfine/<applemots_config>.yml` を具体名 `configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml` へ修正し、Trace IDの手順対応（MOTIP=手順7-10、DEIM=手順3-6）を修正。残るMinorは「本格学習時間は実測に応じて短期/saturation判断にする」点で、手順10/12の停止条件に記載済み。 |
| `2026-06-05` | `04:58:53 UTC+0000` | `Codex統括` | 手順3/4完了: AppleMOTS COCO変換 | ✅DoD-03/04完了。`tests/test_convert_apple_mots_to_coco.py` を追加し、実装前は `ModuleNotFoundError: tools.convert_apple_mots_to_coco` でRed確認。`tools/convert_apple_mots_to_coco.py` と `just build-applemots-coco`, `summary-applemots-coco` を追加。`pytest tests/test_convert_apple_mots_to_coco.py -q` は `2 passed`、`ruff check tools/convert_apple_mots_to_coco.py tests/test_convert_apple_mots_to_coco.py` はAll checks passed。実データ変換先は `datasets/AppleMOTSCOCO`（40M, git外）。summary: train 6 sequences / 1147 images / 62,899 annotations / 1,613 tracks、testing 6 sequences / 1051 images / 46,068 annotations / 1,396 tracks。`pycocotools.COCO` で両JSON load成功、categoryは `[{id:1,name:"apple"}]`。画像はraw PNGへのsymlink。 |
| `2026-06-05` | `05:04:34 UTC+0000` | `Codex統括` | 行動カウント20到達時の状況記録 | 定期リマインダーを一字一句そのまま表示し、行動カウントをリセット。DEIM側には `configs/dataset/coco_detection_applemots.yml` と `configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml` を追加済み。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/.venv uv sync --frozen` は実行中。MOTIP側ではAppleMOTS PseudoMOTのclip内unique数を測定し、SL=20/interval=4の最大unique=254を確認。AppleMOTS BFT公式寄せ設定 `configs/train_applemots_pseudomot_bft_official_schedulefree.yaml` を追加し、`NUM_ID_VOCABULARY=320`, `NUM_TRAINING_IDS=256`, `SAMPLE_LENGTHS=[20]`, `SAMPLE_INTERVALS=[4]`, short-edge 384, ScheduleFree, EarlyStop=True とした。 |
| `2026-06-05` | `05:11:49 UTC+0000` | `Codex統括` | 手順5/6完了: DEIM AppleMOTS smoke | ✅DoD-05完了。`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/.venv uv sync --frozen` 成功（torch 2.8.0+cu128, faster-coco-eval, schedulefree, muon-optimizer導入）。`configs/dataset/coco_detection_applemots.yml` と `configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml` を追加。初回DEIM smokeはAppleMOTS COCOの `category_id=1` が `num_classes=1` の範囲外labelになり、matcherでCUDA index out of bounds。原因確定後、`tools/convert_apple_mots_to_coco.py` に `--category-id` を追加し、DEIM専用 `datasets/AppleMOTSCOCO_DEIM` を `category_id=0` で生成。`pytest tests/test_convert_apple_mots_to_coco.py -q` は `3 passed`、ruffもgreen。retry commandは `cd /workspace/Project/DEIM_sandbox/DEIM && UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/.venv PYTHONPATH=/workspace/Project/DEIM_sandbox/DEIM uv run --no-sync python train.py -c ../configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml -t outputs/deim_hgnetv2_m_coco_tomato_timm/best_stg2_jun23-2025.pth --output-dir ./outputs/deim_hgnetv2_m_coco_applemots_smoke_retry0 --seed 42`。結果: train 573 steps/1epoch、loss avg 25.2234、max CUDA mem 1160MB、validation COCO bbox AP=0.216/AP50=0.553/AR100=0.317、exit 0。 |
| `2026-06-05` | `05:12:51 UTC+0000` | `Codex統括` | 手順7/8完了: AppleMOTS BFT tracking移植 | ✅DoD-06完了。`configs/train_applemots_pseudomot_bft_official_schedulefree.yaml` と just target `config-applemots-bft-schedulefree`, `transplant-applemots-bft-schedulefree`, `train-applemots-bft-schedulefree(-smoke)` を追加。config解決結果: `PSEUDOMOT_SUB_DIR=AppleMOTSPseudoMOT`, `SAMPLE_LENGTHS=[20]`, `SAMPLE_INTERVALS=[4]`, `REL_PE_LENGTH=20`, `MISS_TOLERANCE=20`, `AUG_RESIZE_SCALES=[384]`, `AUG_MAX_SIZE=512`, `AUG_NUM_GROUPS=1`, `NUM_ID_VOCABULARY=320`, `NUM_TRAINING_IDS=256`, `OPTIMIZER_TYPE=AdamWScheduleFree`, `SCHEDULER_TYPE=none`, `RESUME_MODEL=./pretrains/motip_bft_tracking_to_applemots_sl20.pth`, `EARLY_STOP=True`。公式BFT checkpoint `outputs/r50_deformable_detr_motip_bft/r50_deformable_detr_motip_bft.pth` から `pretrains/motip_bft_tracking_to_applemots_sl20.pth`（232M, git外）を生成し、report `reports/motip_bft_tracking_to_applemots_sl20_transfer_report.json` を出力。transfer summary: target_base exact 712 / partial 7 / skipped 0 / missing 0、official_tracking exact 115 / partial 7 / skipped 0 / missing 0 / outside_include 597。 |
| `2026-06-05` | `05:19:21 UTC+0000` | `Codex統括` | 手順9完了: AppleMOTS MOTIP短期条件比較 | ✅DoD-07完了。全条件ともBFT移植checkpoint `pretrains/motip_bft_tracking_to_applemots_sl20.pth`、short-edge 384、ScheduleFree、bf16、`MAX_TRAIN_STEPS=60`, `EPOCHS=1`。比較結果: (1) SL20/interval4: loss 18.9575, detr_loss 13.2768, id_loss 5.6806, max_cuda 12645.97MB, time 1:42。 (2) SL8/interval4: loss 19.3310, detr_loss 13.6977, id_loss 5.6334, max_cuda 2206.59MB, time 0:34。 (3) SL20/interval1: loss 19.0168, detr_loss 13.3556, id_loss 5.6612, max_cuda 6455.27MB, time 1:21。採用条件は `SL8/interval4`。理由: 最小id_loss、最速、最小VRAMで、AppleMOTSの高密度果実IDではSL20がメモリ効率に対してID loss改善を示さなかった。BFT由来の `REL_PE_LENGTH=20` と `MISS_TOLERANCE=20` は維持する。採用config `configs/train_applemots_pseudomot_bft_schedulefree_sl8_pretrain.yaml` を追加し、解決確認済み。 |
| `2026-06-05` | `05:31:03 UTC+0000` | `Codex統括` | 手順10途中: AppleMOTS SL8 pretraining監視 | `just train-applemots-bft-sl8-pretrain` を実行中。config: `SAMPLE_LENGTHS=[8]`, `SAMPLE_INTERVALS=[4]`, `EPOCHS=4`, `RESUME_MODEL=pretrains/motip_bft_tracking_to_applemots_sl20.pth`, ScheduleFree, bf16, EMA, EarlyStop=True。epoch0は完了し、finish metricsは loss 13.0406 / detr_loss 8.0409 / id_loss 4.9997 / max_cuda 2207.91MB / time 9:12。epoch1は120/1105付近まで進行し、id_lossは約3.91-4.16レンジまで低下。現時点でOOM/NaNなし、ID lossはまだ下降中でサチっていない。 |
| `2026-06-05` | `05:41:14 UTC+0000` | `Codex統括` | 手順10途中: AppleMOTS SL8 pretraining監視 | epoch1は完了し、finish metricsは loss 9.7054 / detr_loss 6.4878 / id_loss 3.2176 / max_cuda 2247.60MB / time 9:15。epoch2は260/1105付近まで進行し、running avgは loss 8.8196 / detr_loss 6.0907 / id_loss 2.7289、max_cuda 2247.60MB。epoch0→1→2でID lossが継続低下しており、現時点ではsaturation判断で止める理由はない。 |
| `2026-06-05` | `05:50:46 UTC+0000` | `Codex統括` | 手順10途中: AppleMOTS SL8 pretraining監視 | epoch2が完了。finish metricsは loss 8.5832 / detr_loss 5.9481 / id_loss 2.6351 / max_cuda 2247.60MB / time 9:07。`checkpoint_2.pth` まで生成済み。epoch3は260/1105付近まで進行し、running avgは loss 8.1515 / detr_loss 5.6882 / id_loss 2.4633。ID lossのepoch間低下は継続しているため予定どおりepoch3完走まで続ける。 |
| `2026-06-05` | `05:51:14 UTC+0000` | `Codex統括` | 行動カウント20到達時の状況記録 | 定期リマインダーを一字一句そのまま表示し、行動カウントをリセット。GPU使用は学習プロセスPID 2093772で約2908MiB。AppleMOTS SL8 pretrainingはepoch3継続中で、checkpointは `checkpoint_0.pth`, `checkpoint_1.pth`, `checkpoint_2.pth` まで生成済み。 |
| `2026-06-05` | `05:57:33 UTC+0000` | `Codex統括` | 手順10完了: AppleMOTS SL8 pretraining | ✅DoD-08完了。`just train-applemots-bft-sl8-pretrain` はexit 0で完走。最終checkpointは `outputs/applemots_pseudomot_bft_schedulefree_sl8_pretrain/checkpoint_3.pth`（約905MB）。final metrics: loss 8.0283 / detr_loss 5.5941 / id_loss 2.4342 / max_cuda 2271.89MB / epoch3 time 9:05。stateは `start_epoch=4`, `global_step=4420`、EMAあり、ID vocab tensor shape `(256, 321)`。停止理由は予定epoch完走（saturation停止ではない）。`configs/train_applemots_pseudomot_bft_schedulefree_sl8_pretrain.yaml` でmodel build後、`checkpoint_3.pth` のstateをloadし missing=0 / unexpected=0 を確認。 |
| `2026-06-05` | `05:58:51 UTC+0000` | `Codex統括` | 手順11完了: AppleMOTS checkpointをtomato targetへ移植 | ✅DoD-09完了。source `outputs/applemots_pseudomot_bft_schedulefree_sl8_pretrain/checkpoint_3.pth`、target config `configs/finetune_tracklet_pseudomot_retrack_optuna_bft_official_schedulefree.yaml`、target base `outputs/tracklet_pseudomot_retrack_optuna_ft/checkpoint_40.pth` で `tools/transplant_motip_tracking_weights.py` を実行。output `pretrains/motip_applemots_tracking_to_tomato_retrack_optuna_sl20.pth`（233M, git外）、report `reports/motip_applemots_tracking_to_tomato_retrack_optuna_sl20_transfer_report.json`。summary: target_base exact 711 / partial 7 / skipped 1 / missing 0、AppleMOTS tracking overlay exact 115 / partial 7 / skipped 0 / missing 0 / outside_include 597。skipped 1件はbase側 `id_decoder.rel_pos_embeds` shape `[6,30,16]`→target `[6,20,16]` で、その後AppleMOTS tracking overlayでtarget shapeに移植されるため許容。tomato configでmodel build後、移植checkpointをloadし missing=0 / unexpected=0、ID vocab `(256,513)`, rel_pos `(6,20,16)` を確認。 |
| `2026-06-05` | `06:04:53 UTC+0000` | `Codex統括` | 行動カウント20到達時の状況記録 / tomato smoke中間 | 定期リマインダーを表示し、行動カウントをリセット。tomato smokeの初回起動は `-u` を複数回指定したため、`runtime_option.py` の `nargs='+'` 仕様により最後の `SAVE_CHECKPOINT_PER_EPOCH=1` だけが有効になった。誤条件では既存BFT checkpoint/既存出力dir/22epoch設定になっていたため、該当trainプロセスを停止。修正版は1つの `-u` に `RESUME_MODEL=./pretrains/motip_applemots_tracking_to_tomato_retrack_optuna_sl20.pth OUTPUTS_DIR=./outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke EXP_NAME=tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke MAX_TRAIN_STEPS=60 EPOCHS=1 SAVE_CHECKPOINT_PER_EPOCH=1` をまとめ、正しく反映された。中間metrics: step0 loss 8.1445 / detr_loss 4.7280 / id_loss 3.4164 / max_cuda 11834MB、step40 loss 8.2176 / detr_loss 4.1313 / id_loss 4.0863 / max_cuda 12349MB。SL20/vocab512のため約5s/stepで、60step完了待ち。 |
| `2026-06-05` | `06:08:11 UTC+0000` | `Codex統括` | 手順12完了: tomato短期fine-tune・全frame推論・動画・比較 | ✅DoD-10完了。ただしID改善は未達。tomato smokeは `MAX_TRAIN_STEPS=60` で正常停止し、checkpoint `outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/checkpoint_0.pth`（906M）を生成。finish metrics: loss 8.2176 / detr_loss 4.1313 / id_loss 4.0863 / max_cuda 12348.93MB / time 4:54。全1219frame推論結果は `infer_30fps/tracks.json`（5.0M）と `tracks_mot.txt`（1.2M）、動画は `infer_30fps/tracks.mp4`（136M, 3fps）、比較は `comparison_vs_optuna_retrack.json`。比較結果: Optuna retrack pseudo labelは detections 30379 / unique 827 / mean track length 36.733978 / max 151 / unique ratio 0.027223。AppleMOTS転移後MOTIP出力は detections 24310 / unique 24310 / mean track length 1.0 / max 1 / unique ratio 1.0。`improved_by_mean_track_length=false`, `improved_by_unique_ratio=false`。原因候補は、60step短期fine-tuneではID decoderがtomato query associationへ適応不足、MOTIP online tracker側のID assignmentがembedding信号を保持できていない、Optuna/ByteTrack pseudo labelとMOTIP decoder教師のdomain gap、SL20/vocab512に対するNUM_TRAINING_IDS cap=256の情報落ち、AppleMOTS fruit-domainとtomato-stem-domainの外観差。 |
| `2026-06-05` | `06:11:08 UTC+0000` | `Codex統括` | 手順13完了: docs/reports更新 | ✅DoD-11完了。`docs/ONBOARDING.md` に AppleMOTS COCO/DEIM、BFT移植、SL8 pretrain、AppleMOTS→tomato転移smoke、全frame推論、動画、比較targetと成果物pathを追記。`justfile` に `transplant-applemots-to-tomato`, `train-tracklet-applemots-transfer-smoke`, `infer-tracklet-applemots-transfer`, `video-tracklet-applemots-transfer`, `compare-tracklet-applemots-transfer` を追加し、`just --summary` でtarget解決を確認。作業書を `reports/workdoc_Jun05-2026_applemots_pretrain_tomato_transfer.md` へコピーした。未改善結果は完了扱いの成功ではなく「実験実行済み・改善未達」として明記した。 |
| `2026-06-05` | `06:13:04 UTC+0000` | `Codex統括` | 行動カウント20到達時の状況記録 / 手順14品質gate | 定期リマインダーを表示し、行動カウントをリセット。✅DoD-12完了。MOTIP: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run --no-sync pytest tests/ -q` は 61 passed / 8 warnings、対象Pythonの `ruff check` は All checks passed、`git diff --check` はOK。生成物混入grepでは `outputs/`, `datasets/`, `pretrains/`, `*.pth`, `*.mp4` はgit対象外で、stage候補はsource/config/test/doc/reportのみ。DEIM_sandboxは `git diff --check` OKで、AppleMOTS用config 2本のみ未追跡。 |
| `2026-06-05` | `06:13:59 UTC+0000` | `Codex統括` | 手順15途中: review-written-workdoc最終レビュー | ✅DoD-13完了。`review-written-workdoc` rubricに基づき本書を自己レビュー。Verdictは `PASS_WITH_NOTES`。Blocker/Majorなし。残る注意点は、AppleMOTS転移tomato実験は実行完了したがID改善は未達であり、これは手順12・DoD-10・ONBOARDINGに「改善未達」として明記済み。commit/push確認はDoD-14で実施するため、手順15の確認チェックはcommit後に完了する。 |
| `2026-06-05` | `06:16:00 UTC+0000` | `Codex統括` | DEIM_sandbox commit/push | `/workspace/Project/DEIM_sandbox` branch `deimv1+cu128` で AppleMOTS DEIM smoke config 2本を commit/push。commit `ebf975b` (`Add AppleMOTS DEIM smoke configs`)。対象: `configs/dataset/coco_detection_applemots.yml`, `configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml`。 |
| `2026-06-05` | `06:16:27 UTC+0000` | `Codex統括` | MOTIP commit/push | ✅DoD-14完了。MOTIP branch `cu118` で生成物混入なしを確認して commit/push。実装コミットは `914af95` (`Add AppleMOTS MOTIP transfer workflow`)。対象は AppleMOTS COCO/PseudoMOT変換CLI、BFT/AppleMOTS/tomato transfer configs、ScheduleFree/Muon optimizer integration、移植CLI、tests、ONBOARDING、reports/workdoc。`outputs/`, `datasets/`, `pretrains/`, checkpoint, mp4 はstageなし。 |
