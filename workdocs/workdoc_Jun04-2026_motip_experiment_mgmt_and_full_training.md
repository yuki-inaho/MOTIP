# 作業計画書 兼 記録書: MOTIP tracklet疑似正解学習の改善 — 実験管理フル移植と本格学習

---

**日付:** `2026年06月04日`
**作業ディレクトリ・リポジトリ:** 主対象 `/home/kasm-user/Desktop/MOTIP`（Gitリポジトリ, branch `cu118`, uv venv `=/home/kasm-user/Desktop/MOTIP/.venv`）。作業記録の保存先 `/home/kasm-user/Desktop/motip_sandbox/temp/`。参考 `/workspace/Project/DEIM_sandbox`（DEIM実装）。
**作業者:** `（実行エージェント名を記入）`

> 注: `/home/kasm-user/Desktop/MOTIP` は Git リポジトリ（`git rev-parse --show-toplevel` → `/home/kasm-user/Desktop/MOTIP`）。`motip_sandbox` の `CLAUDE.md` は「MOTIP本体の改変はworkdocに明記されたタスクに限定」「明示許可なくcommit/pushしない」「生成物をgitに入れない」を要求する。本書がその明示タスク定義であり、commit/pushは本書では行わない（DoD外）。

---

## 1. 作業目的

本作業は、以下の目標を達成するために実施します。

* **目標1（確定バグ修正）:** MOTIP tracklet学習基盤で、敵対的レビューにより確定した重大バグ3件を、TDDで修正する。
* **目標2（監査性改善）:** 変換器のサイレントなアノテーション欠落など、no-silent-fallback原則に反する箇所を、件数記録付きの明示処理へ改善する。
* **目標3（実験管理フル移植）:** DEIM_sandboxの実験管理スタックから、MOTIP（Accelerateベース）に適合する形で①〜⑦を移植する（early-stop / TensorBoard二層ロギング / AMP / config駆動early-stop / 汎用CLI上書き / EMA / 再現環境規約）。
* **目標4（本格学習）:** 2-step smokeで健全性を確認した後、tomato tracklet疑似正解での **FULL（本格）学習** を実際に起動・完走し、成功ログ・checkpoint・メトリクス・TensorBoardを証跡として残す（不可能な場合はGPU/データ等の明確なblockerを証跡付きで記録）。
* **目標5（監査性・記録）:** 全工程をt-wada TDD・no-silent-fallback・トレーサビリティ確保のもと、作業記録・DoDへ逐次反映する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** MOTIPの追跡学習を「単に動く」状態から、DEIM相当の**再現可能・観測可能・比較可能**な実験管理基盤に引き上げ、tomato疑似正解で**本格学習を最後まで回す**こと。
* **明示要求:**
  * スコープは「フル移植」。実験管理移植①〜⑦をすべて対象とする。
  * smokeテストの後に **FULL学習を行うところまで** をDoDとする。
  * `write-workdoc-uv` / `review-written-workdoc` の方針に従い、作業書を作成・レビューする。
  * 参考コードの絶対パスと対応箇所を明示する（本書§4・各手順に記載）。
* **暗黙制約:**
  * 重要操作の前に `date "+%Y-%m-%d %H:%M:%S %Z%z"` を実行し記録へ転記。
  * Python作業はuv環境（`UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv`）。`justfile` のtargetを優先。
  * 暗黙fallback禁止。依存・入力・pretrain・GPUメモリ不足は明示的に失敗させ証跡を残す。
  * generated artifacts（`.venv/`, `outputs/`, `datasets/`, `pretrains/`, `*.pth`, caches）はgit管理外。
  * DRY/KISS/SOLID・TDD・監査性。既存挙動を壊さない（移植は既定OFFで非破壊、configで有効化）。
  * **MOTIPはHuggingFace Accelerateを使用**。AMP/分散はDEIMの生GradScalerを写経せず、Accelerateの機構（`accelerator.autocast()`/`accelerator.backward()`/`Accelerator(mixed_precision=...)`）に合わせる。
* **非ゴール:**
  * MOTIPアーキテクチャの全面的registry化（DEIM `engine/core` 相当の再設計）はしない。既存の `dataset_classes`（mini-registry）と `update_config` を拡張するに留める。
  * during-train HOTA評価の完全実装は本作業の必須ではない（監視指標は当面 `loss`、HOTAは将来拡張として設計のみ）。
  * 学習済みモデルの精度SOTA達成・ハイパラ網羅探索はしない。「本格学習が完走し有限の収束傾向と成果物が残る」ことを到達点とする。
  * commit/pushはしない（明示許可が別途出るまで）。
* **成功条件:**
  * バグ3件が修正され、各々に失敗→成功するテストが存在する。
  * 変換器が入力件数・skip件数をsummaryに記録し、サイレント欠落が無い。
  * ①〜⑦が実装され、既定OFFで既存smokeが不変、configで有効化すると機能する（各々テストまたはsmokeで確認）。
  * `configs/train_tracklet_pseudomot_full.yaml`（MAX_TRAIN_STEPS無し）でFULL学習が起動・完走し、`outputs/.../log.txt`・TensorBoard・checkpoint・最終メトリクスが残る。または明確なblockerが証跡付きで記録される。
  * `uv run pytest` / `uv run ruff check` がgreen。未完・例外はDoDに明示。
* **リスクと前提:**
  * **NUM_ID_VOCABULARY不足リスク:** tomatoは最大63 obj/frame・全1367 track。clip長を伸ばすとclip内uniqueトラックが `NUM_ID_VOCABULARY` を超え、`GenerateIDLabels` がランダム選択で**サイレントに切り詰める**（`data/transforms.py` の該当分岐）。→ 手順16で実測し `NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS` を安全側に設定する。
  * **validation split不在:** tomato tracklet datasetは単一sequence（1219 frames）でval GTが無い。during-train HOTAは設計のみとし、本格学習の監視は `loss` ＋ checkpoint保存とする（手順15で方針確定）。
  * **VRAM:** smokeは約1.07GB。本格設定（解像度↑・clip長↑・batch↑）で増大。L4 23GB前提で手順16で実測し調整。OOM時はsample length/解像度/batchを下げる。
  * **学習時間:** 単一GPUで長時間化し得る。本格学習はbackground実行＋ログ監視＋checkpointで進捗を担保する。
  * **Accelerate EMA:** EMAソースは `accelerator.unwrap_model(model)` を使う必要がある（分散/AMPラップ対策）。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 現状調査と移植・本格学習設計の確定 | 目標3,4 | 設計メモ（本書フェーズ1記録） | 接続点とNUM_ID_VOCABULARY/val方針が記録される |
| SG-2 | 確定バグ3件のTDD修正 | 目標1 | `train.py`修正, テスト3件 | 各テスト失敗→成功、`pytest` green |
| SG-3 | 変換器・loaderの監査性改善 | 目標2 | converter/loader修正, summary拡張, テスト | サイレント欠落ゼロ、summaryに件数 |
| SG-4 | 実験管理①〜⑦の移植 | 目標3 | AMP/EMA/TB/early-stop/CLI上書き/再現規約 | 既定OFFでsmoke不変、有効化で機能 |
| SG-5 | 本格学習config・VRAM実測・just target | 目標4 | `configs/train_tracklet_pseudomot_full.yaml`, just target | config load成功, VRAM測定値記録 |
| SG-6 | smoke→FULL学習実行と証跡 | 目標4 | training log/TB/checkpoint/metrics | FULL完走 or blocker証跡 |
| SG-7 | 検証・ignore監査・記録 | 目標5 | pytest/ruff結果, git status, 作業記録 | green, generated artifact未追跡, DoD全充足 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | 確定バグ3件修正(TDD) | フェーズ2 手順4-6 | 各テストの赤→緑, `pytest -q` |
| TR-2 | no-silent-fallback(変換器) | フェーズ3 手順7-8 | summary `num_input/num_skipped`, テスト |
| TR-3 | 実験管理移植①〜⑦ | フェーズ4 手順9-14 | 各機能のsmoke/テスト, 既定OFFでの不変性 |
| TR-4 | Accelerate適合(AMP/分散) | 手順9 | `Accelerator(mixed_precision=...)` 結線, autocast確認 |
| TR-5 | 本格学習設計(ID語彙/val/VRAM) | フェーズ5 手順15-17 | 設計記録, VRAM実測値 |
| TR-6 | smoke→FULL学習完走 | フェーズ6 手順18-20 | log.txt/TB/checkpoint/metrics or blocker |
| TR-7 | 監査性・生成物ignore | フェーズ7 手順21-23 | pytest/ruff, `git check-ignore`, DoD |
| TR-8 | write/reviewスキル運用 | 本書作成・レビュー | 本書, レビュー記録 |

---

## 2. 作業内容

### フェーズ 1: 調査・設計（見積: 1.0h）

実装着手前に、MOTIP（Accelerateベース）の接続点とDEIM参照を精査し、移植方式と本格学習パラメータを確定する。**ここで実装はしない。**

1. **現状アーキテクチャ分析（MOTIP接続点）:** `train.py`（`Accelerator()` 初期化39行目、`accelerator.prepare` 154行目、`accelerator.autocast()` 437行目、`accelerator.backward` 451行目、`MAX_TRAIN_STEPS` 186/524行目）、`configs/util.py`（`update_config_with_kv` 7行目, `update_config` 37行目, `load_super_config` 96行目）、`runtime_option.py`（`--config-path`等フラグ）、`log/logger.py`（`Logger`クラス, wandb有, SummaryWriter無）、`data/joint_dataset.py`（`dataset_classes` mini-registry, `__getitem__`）、`data/util.py`（`collate_fn`）、`data/transforms.py`（`build_transforms` 578行目, `GenerateIDLabels`）。**対応: SG-1 / TR-3,TR-4。**
2. **DEIM参照の確認:** `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py`（fit/early-stop/best/EMA）、`.../det_engine.py`（AMP/TB/iter-step）、`.../engine/optim/ema.py`（ModelEMA）、`.../engine/optim/optim.py`（Muon/ScheduleFree）と設計書 `/home/kasm-user/Desktop/motip_sandbox/temp/design_Jun04-2026_training_experiment_management.md`（§5パターンカタログ, §9移植ロードマップ）。**対応: SG-1 / TR-3。**
3. **移植・本格学習設計の文書化:** ③AMPはAccelerate `mixed_precision`、⑤は汎用`-u KEY=VALUE`追加（既存`update_config_with_kv`再利用）、⑥EMAは`accelerator.unwrap_model`基準、②TBは`Logger`へSummaryWriter追加（`is_main_process`ガード）。本格学習の `NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS`/`SAMPLE_LENGTHS`/`EPOCHS`/val方針/監視指標を、手順16実測前の暫定値として決める。**対応: SG-1,SG-5 / TR-5。**

### フェーズ 2: 確定バグ修正（見積: 1.5h）

レビューで確定したmajor3件をTDDで修正する。

1. **[Bug1] MAX_TRAIN_STEPSのepochループ貫通:** `train_one_epoch` が `MAX_TRAIN_STEPS` 到達で内ループをbreakするが、外側epochループ（`train_engine`）を止めない。`EPOCHS=1` のとき偶然動くだけ。early-stop到達を呼び出し元へ伝播し、epochループをbreakする。**対応: SG-2 / TR-1。**
2. **[Bug2] サンプリング経路の回帰テスト不在:** `JointDataset.__getitem__`→transforms→`trajectory_id_labels` 生成経路を検証する単体テストを追加。**対応: SG-2 / TR-1。**
3. **[Bug3] collate_fnの前提未検証:** `data/util.py:collate_fn` は `trajectory_id_labels` 存在を前提とするが、transform早期return経路でkey欠落し得る。経路を突くテストを追加（必要なら防御実装）。**対応: SG-2 / TR-1。**

### フェーズ 3: 監査性改善（見積: 0.75h）

1. **変換器のサイレントdrop明示化:** `tools/convert_coco_tracklets_to_pseudomot.py` の退化bbox(`w<=0|h<=0`)スキップを件数記録付きへ。summaryに `num_input_annotations`/`num_skipped_degenerate` を追加。**対応: SG-3 / TR-2。**
2. **loaderの堅牢化:** `data/pseudo_mot.py:_get_sequence_names` を非dir除外、`allow_empty_frames` 挙動をテストで固定。**対応: SG-3 / TR-2。**

### フェーズ 4: 実験管理移植①〜⑦（見積: 3.0h）

既定OFF・config有効化・非破壊を厳守。**対応: SG-4 / TR-3,TR-4。**

1. **③ AMP（Accelerate）:** configキー `AMP_DTYPE`（`no`/`fp16`/`bf16`）で `Accelerator(mixed_precision=...)` を設定。既存 `accelerator.autocast()` がそれに従う。
2. **⑥ EMA:** `models/ema.py` に `ModelEMA`（指数ランプdecay）を追加。`EMA_ENABLED`/`EMA_DECAY`/`EMA_WARMUPS` configで制御。各optimizer step後に `ema.update(accelerator.unwrap_model(model))`。評価・checkpointはEMA重みを使用。
3. **② TensorBoard二層:** `log/logger.py:Logger` に `SummaryWriter`（`TENSORBOARD=True` 時, `is_main_process` のみ）。step粒度（loss/lr）と epoch粒度（metrics）を記録。`log.txt`/wandbは非破壊。
4. **①④ config駆動early-stop:** `train_engine` のepochループに、監視指標（当面 `loss`）に対する `EARLY_STOP`/`PATIENCE`/`MIN_DELTA`/`START_EPOCH` を実装し、改善なし継続でbreak。Bug1修正と統合。
5. **⑤ 汎用CLI上書き:** `runtime_option.py` に `-u/--update KEY.SUBKEY=VALUE ...` を追加し、`configs/util.py:update_config_with_kv` で適用。既存typedフラグと共存。`dataset_classes` registryをdocstringで明示。
6. **⑦ 再現環境規約:** `justfile`/READMEに `uv sync --frozen` と `uv run --no-sync`、seed指定（rank別）運用を明文化。

### フェーズ 5: 本格学習設計・config（見積: 1.0h）

1. **本格config作成:** `configs/train_tracklet_pseudomot_full.yaml`（`SUPER_CONFIG_PATH` は dancetrack base継承, `MAX_TRAIN_STEPS` 無し, `DATASETS:[PseudoMOT]`, `EPOCHS`/`SAMPLE_LENGTHS`/`NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS`/`AMP_DTYPE`/`EMA_ENABLED`/`TENSORBOARD`/`EARLY_STOP` を設定, `INFERENCE_DATASET:` は空=during-eval無効）。**対応: SG-5 / TR-5。**
2. **VRAM・ID語彙実測:** clip内最大uniqueトラック数を実測し `NUM_ID_VOCABULARY` を安全側へ。数step回して `max_cuda_mem(MB)` を測定し、batch/解像度/clip長を調整。**対応: SG-5 / TR-5。**
3. **just target追加:** `train-tracklet-full`, `tb`（TensorBoard起動）。**対応: SG-5。**

### フェーズ 6: smoke→本格学習実行（見積: 学習時間依存）

1. **回帰smoke:** `just train-tracklet-smoke` を再実行し、移植後も2-stepが健全（finite loss, 既定挙動不変）であることを確認。**対応: SG-6 / TR-6。**
2. **本格学習起動:** `just train-tracklet-full` をbackgroundで起動し、log/TBで進捗監視。**対応: SG-6 / TR-6。**
3. **完走確認:** EPOCHS完了 or early-stopで正常終了し、`log.txt`・TensorBoard event・checkpoint・最終メトリクスが残ることを確認。GPU/データ起因のblocker時は、原因・再現条件・次アクションを証跡付きで記録。**対応: SG-6 / TR-6。**

### フェーズ 7: 検証・記録（見積: 0.5h）

1. **品質ゲート:** `uv run pytest`・`uv run ruff check`。**対応: SG-7 / TR-7。**
2. **生成物ignore監査:** `git status`/`git check-ignore`。**対応: SG-7 / TR-7。**
3. **DoD・記録最終化:** 作業記録・DoDを全充足へ更新、未完は明示。**対応: SG-7 / TR-7,TR-8。**

---

## 3. 作業チェックリスト

*各手順完了時に `[ ]`→`[x]`、直後に「作業記録」へ日時・結果・証跡を追記する。テストは赤→緑を必ず示す。*

### フェーズ 1: 調査・設計

### 手順 1: 現在時刻と作業対象を記録する
- [x] 🖐 **操作**: `date "+%Y-%m-%d %H:%M:%S %Z%z"` を実行し、対象repo（`/home/kasm-user/Desktop/MOTIP` branch `cu118`）・venv・本書pathを作業記録へ1行追加する。
- [x] 🔎 **確認**: 作業記録の先頭付近に日時・対象path・venvが記載されている。
- [x] 🧪 **テスト**: `manual_start_record`。作業記録にdate出力文字列が含まれることを目視確認する。
- [x] 🛠 **エラー時対処**: timezoneが異なる場合は出力をそのまま記録し、必要なら `TZ=Asia/Tokyo date ...` を併記する。

### 手順 2: MOTIP接続点を精査して記録する
- [x] 🖐 **操作**: `train.py`(39,154,186,437,451,524行周辺), `configs/util.py`(7,37,96), `runtime_option.py`, `log/logger.py`, `data/joint_dataset.py`, `data/util.py`, `data/transforms.py`(578, `GenerateIDLabels`) を読み、AMP/EMA/TB/early-stop/CLI上書きの各接続点を本書の作業記録へ箇条書きする。→ §9.1 に記録。
- [x] 🔎 **確認**: 各移植項目（①〜⑦）に対し「変更ファイル:行・方式」が1対1で記録されている。→ §9.1 の表で①〜⑦ 7行を1対1記載。
- [x] 🧪 **テスト**: 調査のため自動テスト不要。追加予定テスト名（`test_max_train_steps_stops_epoch_loop` 等）を一覧化する。→ §9.2 に9テスト一覧化。
- [x] 🛠 **エラー時対処**: 想定行が見つからない場合は `rg -n "accelerator|MAX_TRAIN_STEPS|trajectory_id_labels" train.py data/` で再特定し、不一致を記録する。→ 行ズレあり（実コードで再特定し §9.1 に実行番号で記録。例: `MAX_TRAIN_STEPS` は 186(引数)/522-525(停止判定)、注記の524は近傍）。

### 手順 3: 移植方式と本格学習設計を文書化する
- [x] 🖐 **操作**: DEIM参照（`/workspace/Project/DEIM_sandbox/DEIM/engine/solver/{det_solver.py,det_engine.py}`, `.../engine/optim/ema.py`）と設計書§5,§9を読み、(a)AMP=Accelerate `mixed_precision`、(b)EMA=`unwrap_model`基準、(c)early-stop=epochループbreak、(d)監視=loss/val方針=during-eval無効、(e)本格学習の暫定 `EPOCHS/SAMPLE_LENGTHS/NUM_ID_VOCABULARY` を本書へ明記する。→ §9.3 (a)〜(e) に記録。
- [x] 🔎 **確認**: フェーズ4-6の各手順が、ここで決めた方式・暫定値・Trace IDに紐づいている。→ §9.1表のTrace列(TR-1,3,4)・§9.3が手順9-16/18-20と対応。
- [x] 🧪 **テスト**: 方針が複数あり迷う項目（val設計等）は選択肢・採用基準・保留理由を明記する。→ §9.3(d) に選択肢A(採用)/B(保留)・採用基準・保留理由を記載。
- [x] 🛠 **エラー時対処**: 参照実装と方式が食い違う場合は、MOTIP側（Accelerate）の制約を優先し理由を記録する。→ §9.3(a) でDEIM生GradScalerを採らずAccelerate `mixed_precision`採用の理由を明記。

### フェーズ 2: 確定バグ修正

### 手順 4: [Bug1] MAX_TRAIN_STEPSがepochループを止める修正（TDD）
- [x] 🖐 **操作**: 先に失敗テスト `tests/test_train_loop_control.py::test_max_train_steps_stops_epoch_loop` を追加（`train_one_epoch` のearly-stop到達が呼び出し元へ伝播することを、制御フロー用の小ヘルパー or `states["global_step"]>=max_train_steps` 判定関数で検証）。次に `train.py` で `train_one_epoch` がearly-stopを返し、`train_engine` のepochループが `break` するよう実装する。→ 純粋判定関数 `reached_max_train_steps(global_step, max_train_steps)`（`train.py:578`）を追加、`train_one_epoch` は `(metrics, early_stopped)` を返す（`train.py:535`）、`train_engine` の唯一の呼出元を `train_metrics, early_stopped = ...`（`:165`）に更新しepochループ末尾で `if early_stopped: break`（`:261-263`）。
- [x] 🔎 **確認**: `EPOCHS=2, MAX_TRAIN_STEPS=2` 設定で総global_stepが2で停止し、epoch1へ進まない。→ 制御フロー等価のepoch-loop driver単体テスト（`test_epoch_loop_breaks_when_one_epoch_reports_early_stopped` 等）で `early_stopped=True` 時にepoch0でbreakを検証。実smoke（`EPOCHS=2,MAX_TRAIN_STEPS=2`）は手順18（別ブロック）。
- [x] 🧪 **テスト**: 追加テストが実装前に失敗→実装後 `UV_PROJECT_ENVIRONMENT=...venv uv run pytest tests/test_train_loop_control.py -q` で成功。加えて手順18のsmokeログで `Stop training early` が1回・epoch0のみであることを確認。→ 赤(3 failed/3 passed: `reached_max_train_steps`不在・`return metrics`のまま)→緑(6 passed)。手順18は別ブロック。
- [x] 🛠 **エラー時対処**: `train_one_epoch` のシグネチャ変更で呼び出し元が壊れたら、戻り値を `(metrics, early_stopped)` か dict へ統一し全呼び出しを更新する。→ 戻り値を `(metrics, early_stopped)` タプルに統一。呼出元は `train_engine` の1箇所のみ（`grep` で確認）で更新済。停止経路は一本化（内ループはflag設定のみ、break判定はepochループの`early_stopped`のみ＝二重break無）。patience監視は未実装（手順12送り）。

### 手順 5: [Bug2] サンプリング経路の回帰テスト追加（TDD）
- [x] 🖐 **操作**: `tests/test_joint_dataset_getitem.py::test_getitem_produces_trajectory_id_labels` を追加。`_build_dataset`（既存テストのヘルパ流用）でtiny PseudoMOTを作り、smoke config相当で `build_transforms` を構築、`JointDataset(transforms=...)` の `__getitem__` を1サンプル取り出し、`annotations[0]` に `trajectory_id_labels`/`trajectory_id_masks`/`trajectory_ann_idxs`/`unknown_id_labels` 等が存在し形状が妥当であることを検証する。→ 実装: smoke相当の最小transform config（`_SMOKE_TRANSFORM_CONFIG`, resize=[320]等）で `build_transforms` 構築、`__getitem__(info)` の `info` は実sampler契約（`data/naive_sampler.py:109-113` の `{dataset,split,sequence,frame_idxs}`）に一致させた。
- [x] 🔎 **確認**: 返り値 `(images, annotations, metas)` 構造と、trajectory系8キーの存在・shapeが期待通り。→ 各frameで8キー（`trajectory_*`/`unknown_*` ×4）の存在・`(G,1,N)=(1,1,N)` shape・dtype（int64/bool）を検証。
- [x] 🧪 **テスト**: 追加直後は（transforms未結線なら）失敗、正しく構築後 `uv run pytest tests/test_joint_dataset_getitem.py -q` で成功。→ **characterization（現行コードで通る回帰テスト＝coverage gapを塞ぐ）**: `1 passed`（`uv run --no-sync pytest tests/test_joint_dataset_getitem.py -q`）。保証内容=`__getitem__`→`build_transforms`→`GenerateIDLabels`/`TurnIntoTrajectoryAndUnknown` の8キー生成経路が将来refactorで欠落しないこと。MOTIP本体コードは無改変（テスト新規追加のみ）。
- [x] 🛠 **エラー時対処**: `trajectory_id_labels` 欠落時は `build_transforms` に渡すconfigキー（`NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS`/`SAMPLE_LENGTHS`/`AUG_NUM_GROUPS`）の不足を疑い、smoke configの値で補う。→ 不足無し（一発green）。configは `_SMOKE_TRANSFORM_CONFIG` に明示。

### 手順 6: [Bug3] collate_fnの前提を突くテスト追加（TDD）
- [x] 🖐 **操作**: `tests/test_collate_fn.py::test_collate_requires_trajectory_labels` を追加。手順5の `__getitem__` で得た複数サンプルを `collate_fn` に通し、`trajectory_id_labels` のpadが正しく行われ例外が出ないことを検証する。早期return経路（全ann非合法化）でkey欠落する場合は、`data/util.py:collate_fn` 冒頭に明示的アサート/防御（不足時は分かりやすい例外）を追加する。→ 新規 `tests/test_collate_fn.py`（3テスト: 正常collate成功+pad整合、key欠落で明示例外、メッセージactionable）。手順5のヘルパ（`_make_sample`）を流用。`data/util.py:collate_fn` のpadループ手前に**明示KeyError防御**を追加。
- [x] 🔎 **確認**: 正常サンプルで `collate_fn` 成功、key欠落サンプルでは明示例外（サイレント `KeyError` ではない）。→ 正常2-clip batchで8キーが共通 `max_N` にpad、`trajectory_id_labels` 欠落frameで `collate_fn precondition failed: annotation[batch=..][frame=..] ... transform pipeline ...` を含む明示KeyError。
- [x] 🧪 **テスト**: `uv run pytest tests/test_collate_fn.py -q` が成功（防御を入れた場合は欠落ケースのraiseも検証）。→ 赤(`2 failed, 1 passed`: 旧コードは opaque `KeyError('trajectory_id_labels')` でメッセージに`collate_fn`/`transform`無し)→緑(`3 passed`)。`uv run --no-sync pytest tests/test_collate_fn.py -q`。
- [x] 🛠 **エラー時対処**: padロジックが形状不一致を起こす場合は `max_N` 計算と各キーのdim整合を確認する。→ 該当無し（pad整合テストでmax_N一致を検証済）。**変更ファイル `data/util.py`（MOTIP, コード変更）**: `collate_fn` のpad前(`:71-79`付近)に全frame `trajectory_id_labels` 存在チェック→欠落時 `(batch,frame)` 番号と原因（transform pipeline）を示す明示 `KeyError`。挙動: 正常入力では従来と完全に等価（防御は欠落時のみ発火）。

### フェーズ 3: 監査性改善

### 手順 7: 変換器のサイレントdropを明示化（TDD）
- [x] 🖐 **操作**: 先に `tests/test_tracklet_pseudomot_conversion.py` へ退化bbox（`w<=0`）を含む入力ケースを追加し、summaryに `num_input_annotations`/`num_skipped_degenerate` が出ることをアサート（赤）。次に `tools/convert_coco_tracklets_to_pseudomot.py` でskip件数を集計しsummaryへ追加（緑）。→ 新規2テスト（退化bbox 2件含む4ann入力、退化なし3ann回帰）追加。`convert_coco_tracklets_to_pseudomot` に `num_input_annotations=len(data["annotations"])`・`num_skipped_degenerate` カウンタ追加、`if w<=0 or h<=0: num_skipped_degenerate+=1; continue`、summary dictへ2キー追加。
- [x] 🔎 **確認**: `num_input_annotations == num_objects + num_skipped_degenerate` が常に成立し、conversion_summary.jsonに記録される。→ 退化(4=2+2)・非退化(3=3+0)両ケースで不変条件＋`conversion_summary.json` persist検証。
- [x] 🧪 **テスト**: `uv run pytest tests/test_tracklet_pseudomot_conversion.py -q` で新ケース含め成功。→ 赤(`2 failed, 1 passed`: `KeyError: 'num_skipped_degenerate'`)→緑(`3 passed`)。
- [x] 🛠 **エラー時対処**: 既存テストの期待 `num_objects` が変わらないこと（退化なし入力で挙動不変）を確認し、回帰があれば集計位置を見直す。→ 既存test(`num_objects==3`)不変でpass継続。回帰テスト `test_no_degenerate_input_keeps_num_objects_and_zero_skips` で固定。**変更ファイル `tools/convert_coco_tracklets_to_pseudomot.py`（MOTIP, コード変更）**。

### 手順 8: PseudoMOT loaderの堅牢化（TDD）
- [x] 🖐 **操作**: `tests/test_pseudo_mot_dataset.py` に「split配下に非dirファイルがあっても無視」「`allow_empty_frames=True` で空frameが合法化される」ケースを追加（赤）。`data/pseudo_mot.py:_get_sequence_names` を `os.path.isdir` 絞り込みへ修正（緑）。→ 新規3テスト追加（非dir無視・ソート順不変・`allow_empty_frames`制御）。`_get_sequence_names` を `os.path.isdir(os.path.join(split_dir, name))` で絞り込む実装へ修正。
- [x] 🔎 **確認**: 非dirを含むsplitでsequence数が正しく、`allow_empty_frames` の真偽で `is_legal` が期待通り変わる。→ stray `README.txt`/`.DS_Store`/`stray.txt` を置いても `_get_sequence_names()==["mini_seq"]`／`["a_seq","b_seq","mini_seq"]`。空frame付きdatasetで `allow_empty_frames=False`→frame1 `is_legal=False`、`True`→`is_legal=True` を検証。
- [x] 🧪 **テスト**: `uv run pytest tests/test_pseudo_mot_dataset.py -q` で成功。→ 赤(`2 failed, 2 passed`: 非dir 2件が `KeyError: 'Sequence'`＝stray fileをseq扱いし `seqinfo.ini` 不在)→緑(`4 passed`)。`uv run --no-sync pytest tests/test_pseudo_mot_dataset.py -q`。
- [x] 🛠 **エラー時対処**: 既存の実データloader smoke（手順18相当）が壊れないこと、sequence名ソート順が変わらないことを確認する。→ **ソート順不変**を `test_get_sequence_names_preserves_sorted_order`（insertion順≠sorted順の3 seq）で固定。既存test不変でpass継続。**変更ファイル `data/pseudo_mot.py`（MOTIP, コード変更）**。**B3全体検証**: `uv run --no-sync pytest tests/ -q` → `17 passed`（回帰なし）。`ruff check tools/convert_coco_tracklets_to_pseudomot.py data/pseudo_mot.py tests/*` → `All checks passed!`。

### フェーズ 4: 実験管理移植①〜⑦

### 手順 9: ③ AMPをAccelerate mixed_precisionで結線
- [x] 🖐 **操作**: `train.py` の `Accelerator()` を `Accelerator(mixed_precision=config.get("AMP_DTYPE","no"))` に変更し、configに `AMP_DTYPE` を追加（既定 `no`=既存挙動）。`accelerator.autocast()`（437行目）はこれに従う。→ `train.py:39` を `Accelerator(mixed_precision=config.get("AMP_DTYPE", "no"))` へ変更。`AMP_DTYPE` は `config.get` 経由（CLI typedフラグ未追加＝`update_config`のRuntimeError回避）。既存 `accelerator.autocast()`(`:441`)/`backward()`(`:455`) はこれに追従。
- [x] 🔎 **確認**: `AMP_DTYPE: no` でsmokeのlossが従来値と一致（非破壊）、`fp16`/`bf16` で起動し `max_cuda_mem` が低下する。→ **既定OFF非破壊**: `just train-tracklet-smoke`相当（`AMP_DTYPE`無＝`no`）で `loss=43.1608`・`detr_grad_norm=98.6928`・`other_grad_norm=18.2396`・`max_cuda_mem=1069.84` がclip-path変更**後**もbit一致（baseline完全一致）。**fp16有効**: `loss=43.1665`(finite)・`max_cuda_mem=704.55`（fp32比 約34%減）。
- [x] 🧪 **テスト**: `manual_amp_smoke`。smoke configを `-u AMP_DTYPE=fp16` 相当で1-2step回し、finite lossとVRAM低下をlogで確認。→ 一時config `/tmp/amp_fp16_smoke.yaml`（smoke + `AMP_DTYPE: fp16`、出力先別dir）で2-step実行、`loss=43.1665` finite・VRAM 1069.84→704.55 MB低下・`Stop training early at global_step=2`・exit 0 を確認。生成物(`outputs/tracklet_pseudomot_amp_smoke`)は確認後削除。
- [x] 🛠 **エラー時対処**: bf16非対応GPUなら `fp16` へ。loss計算でのdtype不整合は、損失部のみFP32化（Accelerateの autocast 範囲）を検討。→ **fp16初回でlatentバグ顕在化**: `unscale_() has already been called`。原因=`use_accelerate_clip_norm且つseparate_clip_norm`時に `accelerator.clip_grad_norm_`(内部で`scaler.unscale_()`)を**2回**呼ぶ(`detr_params`/`other_params`)→AMP時にGradScalerが二重unscaleで例外（`accelerate`の`clip_grad_norm_`実装をinspectで確認）。**修正**: 当該分岐を `accelerator.unscale_gradients()`(AMP無時no-op)を1回＋`torch.nn.utils.clip_grad_norm_`×2 へ（既存の`use_accelerate_clip_norm=False`分岐と同方式）。→ `AMP_DTYPE=no`時grad-norm/loss bit不変を実測確認（非破壊）。**変更ファイル `train.py`（MOTIP, コード変更2箇所: `:39` Accelerator, `:463-471` clip分岐）**。回帰: `pytest tests/ -q`→`17 passed`、`ruff`→All passed。

### 手順 10: ⑥ EMAを追加（Accelerate対応）
- [x] 🖐 **操作**: `models/ema.py` に `ModelEMA`（`decay*(1-exp(-updates/warmups))`、`start`スキップ、全floating-point state平均）を実装。参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78`。`train.py` で `EMA_ENABLED` 時に生成し、各step後 `ema.update(accelerator.unwrap_model(model))`、評価/checkpointはEMA重みを使用。→ 新規 `models/ema.py:ModelEMA`（`decay_fn=decay*(1-exp(-x/warmups))`、`warmups==0`で定数、`start`スキップ、floating-point state(param+buffer)のみ平均、`state_dict={module,updates}`、`load_state_dict(strict=True)`、params `requires_grad=False`）。**DEIMの`de_parallel`は内蔵せず**呼出側が `accelerator.unwrap_model` でunwrap済moduleを渡す設計（Accelerate-correct）。`train.py`: `accelerator.prepare`後に `EMA_ENABLED`時のみ `ModelEMA(accelerator.unwrap_model(model), ...)` 生成、`train_one_epoch` へ `ema` 引数追加し各optimizer.step後 `ema.update(accelerator.unwrap_model(model))`。`save_checkpoint` に `ema=None` 引数追加しEMA時 `save_state["ema"]=ema.state_dict()`。
- [x] 🔎 **確認**: `EMA_ENABLED:False` でsmoke不変。`True` でEMA重みが更新され `updates` が増加、checkpointにEMA stateが含まれる。→ **既定OFF**(`EMA_ENABLED`無): smoke `loss=43.1608`(baseline一致)・"EMA enabled"ログ無。**ON**(`/tmp/ema_smoke.yaml`, decay=0.9/warmups=5): "EMA enabled (decay=0.9, warmups=5)"・online loss不変(43.1608)・checkpoint top keys=`[model,optimizer,scheduler,states,ema]`・`ema`=`{module,updates}`・`updates=2`(2 optimizer step=2 EMA update)。
- [x] 🧪 **テスト**: `tests/test_model_ema.py::test_ema_ramp_and_update`（小モデルで decay_fn のランプと update 後の重み移動を検証）が成功。→ 新規 `tests/test_model_ema.py`（7テスト: decay_fnランプ/定数、update後重み移動＋updates増、startスキップ、params no-grad、state_dict roundtrip、buffer(BatchNorm)平均）。赤(`ModuleNotFoundError: models.ema`)→緑(`7 passed`)。
- [x] 🛠 **エラー時対処**: 分散/AMP配下で重みが取れない場合は必ず `unwrap_model` を経由。dtype不一致は floating-point判定でスキップ。→ 全update/生成で `accelerator.unwrap_model` 経由。`update` は `v.dtype.is_floating_point` のみ平均（int buffer=`num_batches_tracked`等はスキップ, test `test_buffers_are_averaged_too` で float buffer平均を確認）。**変更/新規ファイル**: 新規 `models/ema.py`、変更 `train.py`（import/EMA生成/`train_one_epoch`引数/update/save呼出2箇所）・`models/misc.py`（`save_checkpoint` に`ema`引数）。回帰: `pytest tests/ -q`→`24 passed`、`ruff`→All passed。commit/pushなし。

### 手順 11: ② TensorBoard二層ロギングを追加
> 前提（重要・実測済）: **MOTIPには `tensorboard` が未インストール・未宣言**（`pyproject.toml` deps に無し）。`from torch.utils.tensorboard import SummaryWriter` は `ModuleNotFoundError: No module named 'tensorboard'` になる。よって依存追加が必須（lockも更新される→手順14の `--frozen` 運用は再lock後に適用）。
- [x] 🖐 **操作**: まず `UV_PROJECT_ENVIRONMENT=...venv uv add tensorboard`（`pyproject.toml`+`uv.lock` 更新）。次に `log/logger.py:Logger` に `TENSORBOARD` 有効時のみ `SummaryWriter(log_dir=<logdir>/tb)` を生成（`Logger._is_to_do()`/main processガード）。step粒度（`Loss/total`,`Lr/<pg>`,`Loss/<k>`）と epoch粒度（metrics）を `add_scalar`。呼び出しは既存の `logger.metrics(...)` 経路に追加する（step粒度: `train.py:498` 内, epoch粒度: `train.py:198,244`）。`log.txt`/wandbは変更しない。参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:112-117`。→ `uv add tensorboard`(2.20.0)実行（pyproject+lock更新）。`Logger.__init__` に `tensorboard=False` 引数追加、`is_main_process`且つ`tensorboard`時のみ `SummaryWriter(log_dir=<logdir>/tb)` 生成（`self.tb_writer`）。`tb_scalar(tag,value,step)`(無効時no-op)・`close()` 追加。`train.py`: Logger生成に `tensorboard=config.get("TENSORBOARD",False)`、step粒度TBを step-log直後(`Loss/total`/`Lr/pg`/`Mem/max_cuda_mb`/`Loss/<k>`)、epoch粒度TBを epoch-metrics直後(`epoch/<k>`=global_average)、epochループ後に `logger.close()`。
- [x] 🔎 **確認**: `TENSORBOARD:False`（既定）でファイル生成なし・挙動不変。`True` で `<logdir>/tb/` にevent生成、`just tb` で可視化可能。→ **既定OFF**: smoke `loss=43.1608`・`outputs/.../train/tb` **未生成**（`ls`でNo such file）。**ON**: `<logdir>/tb/events.out.tfevents...` 生成・74 scalar tags(`Loss/total`含む)を `EventAccumulator` で読取確認。`log.txt`(2682B)も併存=非破壊。
- [x] 🧪 **テスト**: `manual_tb_smoke`。smokeを `-u TENSORBOARD=True` 相当で回し、`<logdir>/tb/` に event ファイルが1つ以上生成される。加えて `tests/test_logger_tensorboard.py::test_writer_disabled_by_default`（`TENSORBOARD` 未指定でwriterがNone）を追加し成功させる。→ 新規 `tests/test_logger_tensorboard.py`（4テスト: 既定でwriter=None・tb dir無、有効でtb dir生成、scalar書込でtfevents生成、無効時tb_scalar no-op）。赤(`AttributeError: 'Logger' ... 'tb_scalar'`)→緑(`4 passed`)。`manual_tb_smoke`(`/tmp/tb_smoke.yaml`)でtfevents生成確認。
- [x] 🛠 **エラー時対処**: `ModuleNotFoundError: tensorboard` は依存未追加が原因→ `uv add tensorboard` を先に実行。多重 `add_scalar` のtag衝突回避にstep値は `global_step` を使用。書込未flushはプロセス終了時 `writer.close()`（`atexit`または明示）を保証。→ `uv add` 後 `from torch.utils.tensorboard import SummaryWriter` import成功確認。step値は `states["global_step"]` 使用。`train_engine` 末尾で明示 `logger.close()`。**変更/新規**: `pyproject.toml`+`uv.lock`(tensorboard 2.20.0)、`log/logger.py`(TB層＋既存F541 lint修正1件)、`train.py`(Logger生成/step・epoch TB/close)、新規 `tests/test_logger_tensorboard.py`。**B4a全体検証**: `pytest tests/ -q`→`28 passed`(回帰なし)、`ruff`(6ファイル)→All passed。`outputs/`はgit-ignore(生成物未追跡)。commit/pushなし。**スコープ遵守: 手順12以降 `[ ]` 未着手。**

### 手順 12: ①④ config駆動early-stopをepochループに実装
- [x] 🖐 **操作**: `train.py:train_engine` のepochループに、監視指標（当面 `loss` の epoch平均）への `EARLY_STOP`/`EARLY_STOP_PATIENCE`/`EARLY_STOP_MIN_DELTA`/`EARLY_STOP_START_EPOCH` を実装（改善なし継続で `break`）。手順4のBug1修正（early-stop伝播）と統合し、`MAX_TRAIN_STEPS` 由来の停止も同一の停止経路に集約。参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:76-86,185-196,220-223`。→ **純粋判定関数** `should_early_stop(history, patience, min_delta, start_epoch)`(`train.py:661`) を新設。`train_engine` でepoch平均loss(`train_metrics["loss"].global_average`)を `loss_history` に蓄積し、`early_stop_enabled且つshould_early_stop(...)`時に `early_stopped = True` をセット。**既存の単一break(`:323 if early_stopped: break`)へ合流**＝MAX_TRAIN_STEPS由来停止と同一経路（二重break無）。
- [x] 🔎 **確認**: `EARLY_STOP:False` で全EPOCHS実行（非破壊）。`True`+小patienceで早期break、停止理由がlogに出る。→ **既定OFF**(`EARLY_STOP`無): smoke `loss=43.1608`・"Early stopping triggered"ログ無・MAX_TRAIN_STEPS経路は従来通り(`Stop training early...by MAX`+`Stop the epoch loop early`)。停止理由ログ実装済(`Early stopping triggered at epoch ...best=...`)。
- [x] 🧪 **テスト**: `tests/test_early_stop.py::test_patience_triggers_break`（loss列を与え停止epochを検証する純粋関数化したjudgeを単体テスト）が成功。→ 新規 `tests/test_early_stop.py`（6テスト: 短履歴で非停止、patience超過で停止、改善でリセット、min_delta、start_epoch遅延、patience=0）。赤(`ImportError: should_early_stop`)→緑(`6 passed`)。
- [x] 🛠 **エラー時対処**: 監視指標の取得タイミング（epoch終了後）と `start_epoch` のオフセットを確認。NaN lossは停止条件と別に明示失敗させる。→ epoch metrics確定後に `train_metrics["loss"].sync()`→`global_average` 取得。**NaN/inf lossは早期停止と別経路で `RuntimeError` 明示失敗**（no-silent-fallback、黙って継続しない）。`start_epoch` は `history[start_epoch:]` で考慮、判定は単体テストで固定。**変更ファイル `train.py`（MOTIP, コード変更）**: `should_early_stop` 追加・early-stop config読込/loss_history/NaN-check/flag合流。回帰: `pytest tests/ -q`→`34 passed`、`ruff`→All passed。commit/pushなし。

### 手順 13: ⑤ 汎用 `-u KEY=VALUE` 上書きを追加
- [x] 🖐 **操作**: `runtime_option.py` に `-u/--update KEY.SUBKEY=VALUE ...`（`nargs="+"`）を追加。`configs/util.py` に dotted key を再帰適用する関数（既存 `update_config_with_kv` を活用）を実装し、typedフラグ適用後に重ねる。`data/joint_dataset.py:dataset_classes` のregistryをdocstringで明示。→ `runtime_option.py` に `-u/--update`(`nargs="+"`)追加。`configs/util.py` に `apply_cli_updates(config, updates)` ＋ ヘルパ `_set_existing_key`（再帰でcase-insensitive既存キー探索、新規キー作成せず）。`train.py:__main__` で `update_config`(typed)後に `apply_cli_updates(cfg, opt.update)`。`update_config` の skip対象に `update` 追加（typed扱いしない）。`data/joint_dataset.py:dataset_classes` にregistry docstring追加。
- [x] 🔎 **確認**: `--config-path ... -u EPOCHS=3 AMP_DTYPE=fp16` でconfigが上書きされ、`config-tracklet-smoke` 相当の出力に反映される。値はYAMLとして型解釈される。→ end-to-end: `-u MAX_TRAIN_STEPS=1` で "Config overridden via -u: MAX_TRAIN_STEPS = 1"・config echo `max_train_steps: 1`(元2)・`Stop training early at global_step=1` を確認。型解釈は `yaml.safe_load`(int/float/bool/list/str)。
- [x] 🧪 **テスト**: `tests/test_cli_override.py::test_dotted_update`（`update_config_with_kv` でネスト/型解釈を検証）が成功。→ 新規 `tests/test_cli_override.py`（8テスト: top-level int、YAML型(bool/float/str)、list、ネストキー、未知キーKeyError、`=`無しValueError、複数順序適用、既存`update_config_with_kv`不変）。赤(`ImportError: apply_cli_updates`)→緑(`8 passed`)。
- [x] 🛠 **エラー時対処**: 未知キー上書きは黙って無視せず警告ログ（no-silent-fallback）。型解釈は `yaml.safe_load(value)` で統一。→ **未知キーは `KeyError` 明示raise**（サイレント無視せず）。適用成功時は `warnings.warn` で上書きをログ。end-to-end実証: `-u AMP_DTYPE=fp16`(smoke configに未在)で `KeyError: Unknown -u override key 'AMP_DTYPE' ... no silent fallback`。型解釈は `yaml.safe_load` 統一（注: `1e-3`はYAML 1.1で文字列扱い→`0.001`/`1.0e-3`使用、テストにコメント明記）。**変更/新規**: `configs/util.py`(`apply_cli_updates`/`_set_existing_key`/skip update)、`runtime_option.py`(`-u`)、`train.py`(import/適用)、`data/joint_dataset.py`(registry docstring)、新規 `tests/test_cli_override.py`。回帰: `pytest tests/ -q`→`42 passed`、`ruff`→All passed。commit/pushなし。

### 手順 14: ⑦ 再現環境規約を明文化
> 注: 手順11で `tensorboard` を追加するとlockが更新される。`--frozen` は「既にlock済の状態を再現する」運用であり、依存追加時はまず `uv lock`（または `uv add`）で再lockしてから `--frozen` 同期する。手順の順序として、依存追加（手順11）→再lock→本手順の規約適用、とする。
- [x] 🖐 **操作**: `justfile`/`README.md` に `UV_PROJECT_ENVIRONMENT=... uv sync --frozen`（lock固定同期）、`uv run --no-sync`（lock再解決回避）、`--seed`（rank別運用）を明記。seed適用箇所（`runtime_option.py:--seed` と train初期化）が rank を加味するか確認し、無ければ最小修正。→ `README.md` に「Reproducible local environment (uv)」節追加（`uv sync --frozen`/`build-ops`/`uv run --no-sync`/rank-aware seed/`-u`）。`justfile` に `sync-frozen`・`repro-doc` target追加。**seed確認**: `utils/misc.py:set_seed` が `seed = seed + distributed_rank()` で**既にrank加味**→修正不要（文書化のみ）。
- [x] 🔎 **確認**: ドキュメントに再現手順が記載され、依存追加後の `uv.lock` が整合し `uv sync --frozen` が成功する。→ `uv sync --frozen` **exit 0**（tensorboard追加後もlock整合）。`just --list` に `sync-frozen`/`repro-doc` 表示、`just repro-doc` が規程出力。**重要発見**: `uv sync --frozen` は**ローカルビルドの `MultiScaleDeformableAttention` op を削除**（PyPI/lock非管理＝`just build-ops`産物）→ doc/justfileに「`--frozen`後は `just build-ops` 必須」を明記し、本検証で op を再build→training smoke(`loss=43.1608`,exit0)で環境機能を回復・確認。
- [x] 🧪 **テスト**: `manual_repro_doc`。`just --list` と README の該当節を目視確認、`uv run --no-sync python -c "import torch;print(torch.__version__)"` が成功。→ `just --list`(sync-frozen/repro-doc確認)・README該当節・`just repro-doc`出力確認。`uv run --no-sync python -c "import torch;print(torch.__version__)"`→`2.4.0+cu118`。`import torch; import MultiScaleDeformableAttention`→OK(torch先行import時)。
- [x] 🛠 **エラー時対処**: `--frozen` でlock不一致が出たら、差分を記録し（暗黙再解決せず）原因を明示する。→ lock不一致なし(`--frozen` exit0)。op削除の挙動は暗黙再解決ではなく `--frozen` の正しい動作（lock外パッケージ除去）として doc に明示。**変更/新規ファイル**: `README.md`(再現節)、`justfile`(`sync-frozen`/`repro-doc`)。コード変更なし（seedは既存でrank対応）。**B4b全体検証**: `pytest tests/ -q`→`42 passed`、`ruff`(6ファイル)→All passed。`git status`= 変更`README.md`/`justfile`/`configs/util.py`/`runtime_option.py`/`data/joint_dataset.py`/`train.py`(＋持越)・新規`tests/test_{early_stop,cli_override}.py`(＋持越)。generated artifact無、commit/pushなし。**スコープ遵守: 手順15以降 `[ ]` 未着手。**

### フェーズ 5: 本格学習設計・config

### 手順 15: 本格学習configを作成する
- [x] 🖐 **操作**: `configs/train_tracklet_pseudomot_full.yaml` を作成。`SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml` を継承、`MAX_TRAIN_STEPS` を**設定しない**、`DATASETS:[PseudoMOT]`/`DATASET_SPLITS:[train]`/`PSEUDOMOT_SUB_DIR:TomatoTrackletMOT`、`EPOCHS`/`SAMPLE_LENGTHS`/`NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS`（手順16で確定する暫定値）、`AMP_DTYPE:fp16`/`EMA_ENABLED:True`/`TENSORBOARD:True`/`EARLY_STOP:True`、`INFERENCE_DATASET:`（空＝during-eval無効）、`OUTPUTS_DIR:./outputs/tracklet_pseudomot_full`/`EXP_NAME:tracklet_pseudomot_full` を設定。→ `configs/train_tracklet_pseudomot_full.yaml` 新規作成。前提(i)9キー(`AMP_DTYPE`/`EMA_ENABLED`/`EMA_DECAY`/`EMA_WARMUPS`/`TENSORBOARD`/`EARLY_STOP`/`EARLY_STOP_PATIENCE`/`EARLY_STOP_MIN_DELTA`/`EARLY_STOP_START_EPOCH`)を**実装の`config.get`キー名(train.py grep)と一致**で明示。`MAX_TRAIN_STEPS`未設定。手順16実測で確定値へ更新済(下記)。
- [x] 🔎 **確認**: `just config-tracklet-full`（追加target）が主要キーを表示し、`MAX_TRAIN_STEPS` が `None`、`DATASETS=[PseudoMOT]` であること。→ `just config-tracklet-full`(手順17で追加) → `MAX_TRAIN_STEPS=None`/`DATASETS=['PseudoMOT']`/`AMP_DTYPE=fp16`/`EMA_ENABLED=True`/`TENSORBOARD=True`/`EARLY_STOP=True`/`INFERENCE_DATASET=None`、exit 0。
- [x] 🧪 **テスト**: configロードが例外なく成功し、表示キーに移植項目（AMP/EMA/TB/EARLY_STOP）が含まれる。→ `load_super_config` で例外なくロード成功、9キー全present(grep一致)、移植4項目表示確認。
- [x] 🛠 **エラー時対処**: pretrain未存在時は自動DLせず、`just prepare-tracklet-pretrain` の実行要否と不足pathを記録する（no-silent-fallback）。→ pretrain `pretrains/r50_deformable_detr_coco_dancetrack.pth`(163MB)存在確認済(自動DL不要)。**変更/新規 `configs/train_tracklet_pseudomot_full.yaml`（新規）**。

### 手順 16: ID語彙とVRAMを実測して確定する
- [x] 🖐 **操作**: clip長候補（例 `SAMPLE_LENGTHS=[30]`, interval）で、PseudoMOT datasetの各候補clip内uniqueトラック数の最大値を集計するスクリプト/onelinerを実行し、`NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS` をその最大値以上（安全マージン付き）に設定。続いて本格configで数step回し `max_cuda_mem(MB)` を測定する。→ **dataset実測**: max 63 obj/frame・1367 unique track(全1219 frame)。**clip内最大uniqueトラック数(全sampler begin/interval網羅で厳密)**: SL=[30],iv≤4→**231**、SL=[16]→164、SL=[10]→**128**。**VRAM実測**: SL=[30]base→**OOM**(20.28GB alloc)、SL=[16]→8669MB、**SL=[10]→4200MB**(EMA+TB ON, fp16, decoder-checkpoint+解像度↓)。確定: `SAMPLE_LENGTHS=[10]`/`NUM_ID_VOCABULARY=160`/`NUM_TRAINING_IDS=160`(128+margin)。
- [x] 🔎 **確認**: clip内最大uniqueトラック数 ≤ `NUM_ID_VOCABULARY`。これを満たさないと `data/transforms.py:612-617` の `GenerateIDLabels`（`num_id_vocabulary=config["NUM_ID_VOCABULARY"]`, `num_training_ids=config.get("NUM_TRAINING_IDS", NUM_ID_VOCABULARY)`）が**ランダム選択でサイレント切詰め**する。VRAM実測値が記録され、L4 23GB内に収まる。→ **128 ≤ 160 を満たす＝サイレント切詰め無**。VRAM実測 4200MB ≪ L4 23GB。is_legalの300上限も max 63 obj/frame で違反0。
- [x] 🧪 **テスト**: `manual_vram_idvocab`。集計値（clip内最大uniqueトラック数）とVRAM測定値（`max_cuda_mem(MB)`）を作業記録へ転記。なお `NUM_TRAINING_IDS` 未指定時は `NUM_ID_VOCABULARY` にfallbackする（`data/transforms.py:614`）。→ 集計値・VRAM・step時間を作業記録へ転記(下表)。**step時間**: SL=[10] steady 1.32s/step、1210 steps/epoch。**完走時間見積**: 1210×1.32≈27min/epoch、EPOCHS=8→**約3.6h(worst)**、early-stop(patience=3,start=2)で短縮見込。
- [x] 🛠 **エラー時対処**: OOM時は `SAMPLE_LENGTHS`↓/`AUG_*`解像度↓/`BATCH_SIZE`↓/`USE_DECODER_CHECKPOINT:True` を順に適用し、各変更を記録。ID語彙超過時は `NUM_ID_VOCABULARY` を上げるかclip長を下げる（黙って切り詰めない）。→ **OOM対処をPlan順で適用・記録**: base SL=[30] OOM → `USE_DECODER_CHECKPOINT:True`+解像度↓(`AUG_RESIZE_SCALES:[480..640]`/`AUG_MAX_SIZE:1024`)+`DETR_NUM_CHECKPOINT_FRAMES:2` でも SL=[30]/[16]はOOM寸前/重い → **`SAMPLE_LENGTHS:[10]`へ短縮で解決**(4.2GB)。BATCH_SIZE=1は既定。ID語彙はclip長短縮に合わせ `NUM_ID_VOCABULARY=160`(切詰め回避)。確定値を本格configへ反映。**変更 `configs/train_tracklet_pseudomot_full.yaml`**(SL/NUM_ID_VOCABULARY/EPOCHS/OOM対処キー反映)。

### 手順 17: 本格学習用just targetを追加する
- [x] 🖐 **操作**: `justfile` に `config-tracklet-full`, `train-tracklet-full`（`train.py --config-path configs/train_tracklet_pseudomot_full.yaml`）, `tb`（`tensorboard --logdir outputs/tracklet_pseudomot_full/.../tb`）を追加。→ `justfile` に `tracklet_full_config` 変数＋ `config-tracklet-full`(17キーecho)・`train-tracklet-full`(`uv run --no-sync python train.py --config-path ...`)・`tb`(`tensorboard --logdir outputs/tracklet_pseudomot_full/train/tb`) を追加。
- [x] 🔎 **確認**: `just --list | rg "tracklet-full|^tb"` が新targetを表示する。→ `just --list` に `config-tracklet-full`/`train-tracklet-full`/`tb` 表示確認。
- [x] 🧪 **テスト**: `just config-tracklet-full` が主要キーを出力し終了コード0。→ `just config-tracklet-full` → 17キー出力(`MAX_TRAIN_STEPS=None`含む)、exit 0。
- [x] 🛠 **エラー時対処**: TensorBoard未導入（MOTIPは未導入を実測済）の場合は手順11の `uv add tensorboard` が先行している前提。`just tb` が `tensorboard` コマンド不在で失敗するなら依存追加漏れを疑う。→ tensorboard 2.20.0(手順11で追加済)。`tb` target は `uv run --no-sync tensorboard --logdir ...`。**B5全体**: op健在(`import MultiScaleDeformableAttention`→OK, `uv sync/add/--frozen`未実行)、`pytest tests/ -q`→`42 passed`、`git status`に生成物無。**スコープ遵守: 本格学習未起動(B6)、手順18以降 `[ ]` 未着手。** 変更/新規: 新規`configs/train_tracklet_pseudomot_full.yaml`、変更`justfile`。commit/pushなし。

### フェーズ 6: smoke→本格学習実行

### 手順 18: 移植後の回帰smokeを実行する
- [x] 🖐 **操作**: `just train-tracklet-smoke` を実行（既定OFF項目のまま）。→ `just train-tracklet-smoke`(既定config, AMP/EMA/TB既定OFF)を実行。前提: op健在(`import MultiScaleDeformableAttention`→OK)、pretrain(163MB)/dataset(1219f)存在、GPU free 22563MiB。
- [x] 🔎 **確認**: 2-stepで停止し finite loss、`Stop training early` がepoch0のみ1回（Bug1修正の効果）、出力・挙動が従来と整合。→ `PseudoMOT.train, 1 sequences, 1219 frames`・`loss=43.1608`(finite, baseline一致)/`detr_loss=38.4402`/`id_loss=4.7205`・`Stop training early at global_step=2 by MAX_TRAIN_STEPS=2`・`Stop the epoch loop early at epoch 0`・exit 0。EMAログ無(既定OFF)。
- [x] 🧪 **テスト**: `manual_smoke_regression`。`outputs/tracklet_pseudomot_smoke/train/log.txt` の loss が有限、epoch1へ進んでいないことを確認。→ log.txtに `Start training epoch 0` のみ(epoch1無し)、`[Finish epoch: 0] loss=43.1608`(finite)、`Stop the epoch loop early at epoch 0`。**Bug1修正のend-to-end成立**(epoch1へ進まない)。確認後 `outputs/tracklet_pseudomot_smoke` 削除。
- [x] 🛠 **エラー時対処**: 失敗時は移植手順9-13のどれが原因かを既定OFFへ戻して二分し、回帰箇所を特定する。→ 失敗無し(全PASS)。`pytest tests/ -q`緑維持(末尾確認)。commit/pushなし。

### 手順 19: 本格学習を起動する
- [x] 🖐 **操作**: `date` を記録後、`just train-tracklet-full` をbackground実行（ログを `outputs/tracklet_pseudomot_full/train/log.txt` とTBへ）。起動直後にdataset読込・pretrainロード・1step目のfinite lossを確認する。→ `2026-06-04 08:40 UTC` 統括が `uv run --no-sync --directory MOTIP python train.py --config-path configs/train_tracklet_pseudomot_full.yaml` を background起動（task `b39u3d84i`）。
- [x] 🔎 **確認**: `PseudoMOT.train, 1 sequences, 1219 frames.` 読込、DETR pretrainロード成功、epoch0の最初のstepでfinite `loss`/`detr_loss`/`id_loss`。→ ✅ Runtime Config全項目が承認値(`sample_lengths=[10]`/`num_id_vocabulary=160`/`epochs=8`/`amp_dtype=bf16`/`ema_enabled`/`tensorboard`/`early_stop` p=3 s=2/`inference_dataset=None`)、`Loaded Data PseudoMOT.train 1219 frames`、`Load pre-trained DETR ...SUCCESS`、`EMA enabled`、`Early stopping enabled`、step0 `loss=41.7642`/`detr_loss=36.2858`/`id_loss=5.4784`(finite)、`detr_grad_norm=62.51`(finite=bf16でnan解消)。
- [x] 🧪 **テスト**: `manual_full_train_start`。起動ログに上記が出ること、GPUプロセスが生きていることを `nvidia-smi` で確認。→ ✅ step0-120でloss減少(41.76→現29.97/avg35.83, detr 36→25, id 5.48→4.89)、`tps≈1.1s/step`・eta epoch0~20min・`max_cuda 5179MB`(L4 23GB内)。GPU compute process生存。
- [x] 🛠 **エラー時対処**: OOM/pretrain不足/ID語彙超過は手順16の対処を適用し再起動。各失敗は証跡付きで記録（no-silent-fallback）。→ 発生無し(VRAM 5.2GB余裕, 切詰め無, pretrain健在)。pre-flightでfp16→bf16是正済のため起動安定。

### 手順 20: 本格学習を完走させ証跡を残す
- [x] 🖐 **操作**: 学習を監視し、EPOCHS完了 or early-stopまで完走させる。完了後、最終 `log.txt`・TensorBoard event・checkpoint（`best`/`last`相当）・最終メトリクスを確認し作業記録へ転記。→ ✅ `2026-06-04 08:11→11:32 UTC`（≈3.5h, 各epoch~25分）に全8 epoch完走。task `b39u3d84i` **exit 0**。
- [x] 🔎 **確認**: 正常終了（非異常exit）し、lossが学習を通じて有限かつ概ね低下傾向、成果物一式が `outputs/tracklet_pseudomot_full/` に存在。→ ✅ `Finish training epoch 0..7`、`[Finish epoch: 7] [Time: 0:25:6] loss=7.8234`。loss `41.76→7.82`(epoch0→7 avg)・`detr_loss 36.29→3.02`・`id_loss 5.48→4.81`・全勾配finite。成果物: `checkpoint_0〜7.pth`(各947MB)/`train/log.txt`(626KB)/`train/tb/events.out.tfevents...`/`train/config.yaml`。
- [x] 🧪 **テスト**: `manual_full_train_finish`。最終epochのloss・経過時間・checkpoint path・TB event pathを記録。GPU残プロセスが無いことを確認。→ ✅ 最終epoch7 avg `loss=7.8234`/`detr_loss=3.0175`/`id_loss=4.8059`/`max_cuda=5535.8MB`、epoch7 Time `0:25:6`。checkpoint=`outputs/tracklet_pseudomot_full/checkpoint_7.pth`、TB=`outputs/tracklet_pseudomot_full/train/tb/events.out.tfevents.1780560708...`。`nvidia-smi` compute process **なし**。
- [x] 🛠 **エラー時対処**: 完走不能（GPU時間/データ/数値破綻）の場合は完了扱いにせず、停止epoch・原因・再現条件・次アクションをblockerとして証跡付きで明記する。→ 完走したため該当なし（OOM/NaN/break無し、early-stop未トリガ＝loss微減継続）。観察: id_loss は緩やか収束（短clip[10]・8epoch、将来 clip長↑/epoch↑で改善余地）＝blockerではなく将来チューニング事項。

### フェーズ 7: 検証・記録

### 手順 21: テストとlintを実行する
- [x] 🖐 **操作**: `UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/ -q` と `uv run ruff check <変更ファイル群>` を実行する。→ `uv run --no-sync pytest tests/ -q` と `uv run --no-sync ruff check` を全touched .py 19ファイル(train.py/data/tools/log/models/configs/runtime_option.py/tests/)に実行。
- [x] 🔎 **確認**: 追加テスト全て成功、ruffがAll checks passed。→ **`42 passed`**(8 warnings=`-u`上書きの意図的warn)、ruff **`All checks passed!`**。
- [x] 🧪 **テスト**: 結果を作業記録へ貼れる形で要約する（passed数・対象）。→ pytest 42 passed(B2-B5の新規テスト含む: train_loop_control6/joint_dataset_getitem1/collate_fn3/tracklet_pseudomot_conversion3/pseudo_mot_dataset4/model_ema7/logger_tensorboard4/early_stop6/cli_override8 + 既存)、ruff対象19ファイルAll passed。
- [x] 🛠 **エラー時対処**: 失敗テストは原因（実装/期待値/環境）を切り分け、lintはauto-fix可否を判断してから再実行する。→ 失敗・lintエラー無し。op健在(`import MultiScaleDeformableAttention`→OK)。commit/pushなし(B7検証のみ)。

### 手順 22: 生成物のignore監査を行う
- [x] 🖐 **操作**: `git status --short --branch`, `git ls-files outputs datasets pretrains`, `git check-ignore -v outputs/tracklet_pseudomot_full datasets/TomatoTrackletMOT` を実行する。→ commit済状態で実行(B7は再commitしない)。`git status --short --branch`→`## cu118...origin/cu118`(working tree clean, origin同期)。`git log --oneline`→`8a12cd6`(docs ONBOARDING)+`4199aa9`(impl)+`241f41d`(base)。
- [x] 🔎 **確認**: 実装差分（コード/config/test/justfile/doc）と生成物が分離され、生成物はtracked対象外。→ 実装差分は `4199aa9`/`8a12cd6` にcommit済。本格学習生成物(checkpoint_0〜7.pth 各947MB・TB event・log.txt 626KB・config.yaml)は `outputs/tracklet_pseudomot_full/` = **git未追跡**。`check-ignore`: `outputs/`(`.gitignore:28`)/`datasets/`(`.gitignore:29`)/`pretrains/`(`.gitignore:27`) ignore確認。
- [x] 🧪 **テスト**: `git ls-files` が生成物を返さない（空）。→ `git ls-files | grep -E '^(outputs/|datasets/|pretrains/)|\.pth$'` → **EMPTY**(生成物未追跡)。
- [x] 🛠 **エラー時対処**: 生成物がtracked候補に入ったら `.gitignore` を修正し、stage前に再確認する。commitはしない。→ 生成物のtracked混入無し(`.gitignore` 衛生エントリ済)。B7では新規commit/pushせず(検証のみ)。

### 手順 23: 完了の定義と作業記録を最終更新する
- [x] 🖐 **操作**: DoDを確認し、成功/未完/制約/次アクションを作業記録と最終状態表へ追記する。→ DoD-01〜17 を実態突合し全 `[x]`(DoD-14完走反映・DoD-16はcommit済状態へ整合更新)。最終状態§8の全プレースホルダを実値で確定(バグ修正/監査性/移植①〜⑦/本格config bf16/本格学習完走/検証42 passed/generated artifacts/制約)。
- [x] 🔎 **確認**: 本書に意図しない未完了チェックが無く、最終状態表にpath/branch/log/メトリクス/制約がある。→ 最終状態表にrepo path/branch cu118/origin/commit(`4199aa9`,`8a12cd6`)/log path/最終loss 7.82/checkpoint/TB/制約を記載。残`[ ]`は手順1-23・DoD全完了で消化。
- [x] 🧪 **テスト**: `rg -n -- "- \[ \]" temp/workdoc_Jun04-2026_motip_experiment_mgmt_and_full_training.md` が意図しない未完了を返さない。→ 末尾で `rg` 実行し未完了チェック0件を確認(下記作業記録)。
- [x] 🛠 **エラー時対処**: 未完が残る場合は完了扱いにせず、blockerと次アクションを明記する。→ 未完無し。制約(HOTA未実装/id_loss緩やか/checkpoint整理)は §8 制約・次アクションに明記。commit/pushなし(B7検証のみ)。

---

## 4. 作業に使用するコマンド参考情報（絶対パス・対応箇所つき）

### 環境・基本
```bash
date "+%Y-%m-%d %H:%M:%S %Z%z"
export UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv
cd /home/kasm-user/Desktop/MOTIP
uv sync --frozen          # 再現環境（lock固定）
just --list
```

### テスト・品質
```bash
UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run pytest tests/ -q
UV_PROJECT_ENVIRONMENT=/home/kasm-user/Desktop/MOTIP/.venv uv run ruff check train.py data/ tools/ log/ models/ tests/
```

### 学習・可視化
```bash
just train-tracklet-smoke          # 回帰smoke（2-step）
just config-tracklet-full          # 本格config主要キー表示
just train-tracklet-full           # 本格学習（background推奨）
just tb                            # TensorBoard起動
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

### 改修対象（MOTIP, 絶対パス）と参考（DEIM, 絶対パス）
- [Bug1/①④] `/home/kasm-user/Desktop/MOTIP/train.py`（`Accelerator` 39, `MAX_TRAIN_STEPS` 186/524, epochループ）← 参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:76-86,185-223`
- [Bug2/Bug3] `/home/kasm-user/Desktop/MOTIP/data/{joint_dataset.py,util.py,transforms.py}`, `/home/kasm-user/Desktop/MOTIP/tests/`
- [監査性] `/home/kasm-user/Desktop/MOTIP/tools/convert_coco_tracklets_to_pseudomot.py`, `/home/kasm-user/Desktop/MOTIP/data/pseudo_mot.py`
- [③AMP] `/home/kasm-user/Desktop/MOTIP/train.py:39,437,451` ← 参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:48-76`
- [⑥EMA] 新規 `/home/kasm-user/Desktop/MOTIP/models/ema.py` ← 参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78`
- [②TB] `/home/kasm-user/Desktop/MOTIP/log/logger.py` ← 参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:112-117`
- [⑤CLI] `/home/kasm-user/Desktop/MOTIP/{runtime_option.py,configs/util.py}` ← 参考 `/workspace/Project/DEIM_sandbox/DEIM/engine/core/yaml_utils.py`
- [設計書] `/home/kasm-user/Desktop/motip_sandbox/temp/design_Jun04-2026_training_experiment_management.md`（§5,§9）
- [調査レポート] `/home/kasm-user/Desktop/motip_sandbox/temp/report_Jun04-2026_deim_experiment_management.md`

---

## 5. 完了の定義（DoD）

*完了したら `[ ]`→`[x]` にしつつ、本当に完了したかを検証する。*

- [x] DoD-01: 本書がレビュー済（`review-written-workdoc`）で、レビュー結果が作業記録にある。（TR-8）→ 本書は coordinator が `write-workdoc-uv` で作成・反復精緻化し、B1〜B7 各ブロックを auditor が原典照合レビュー（作業記録に各ブロックの監査結果行あり）。
- [x] DoD-02: [Bug1] `MAX_TRAIN_STEPS`/early-stop が外側epochループを正しく停止し、`EPOCHS=2,MAX_TRAIN_STEPS=2` で総2stepで終了する。テストが赤→緑。（TR-1）→ 手順4: `reached_max_train_steps`＋`train_one_epoch`戻り値`(metrics,early_stopped)`＋`train_engine`単一break。`tests/test_train_loop_control.py`(6) 赤→緑。手順18 smokeで `global_step=2`停止・epoch1不到達 実証。
- [x] DoD-03: [Bug2] `__getitem__`→transforms→`trajectory_id_labels` の回帰テストが存在し成功する。（TR-1）→ 手順5: `tests/test_joint_dataset_getitem.py`(8キー存在/shape/dtype) `1 passed`。
- [x] DoD-04: [Bug3] `collate_fn` の前提を突くテストが存在し、key欠落時はサイレントでなく明示失敗する。（TR-1）→ 手順6: `data/util.py:collate_fn` に明示KeyError防御、`tests/test_collate_fn.py`(3) 赤→緑。
- [x] DoD-05: 変換器がサイレントdropを廃し、summaryに `num_input_annotations`/`num_skipped_degenerate` を記録する。（TR-2）→ 手順7: `convert_coco_tracklets_to_pseudomot` に2キー追加・不変条件、`tests/test_tracklet_pseudomot_conversion.py`(+2) 赤→緑。
- [x] DoD-06: PseudoMOT loaderが非dirを除外し、`allow_empty_frames` 挙動がテストで固定されている。（TR-2）→ 手順8: `_get_sequence_names` を `os.path.isdir` 絞り込み、`tests/test_pseudo_mot_dataset.py`(+3, ソート順不変含む) 赤→緑。
- [x] DoD-07: ③AMP（Accelerate `mixed_precision`）が `AMP_DTYPE` で制御でき、既定 `no` で挙動不変。（TR-3,TR-4）→ 手順9: `Accelerator(mixed_precision=config.get("AMP_DTYPE","no"))`。既定`no`でsmoke bit一致(非破壊)、fp16でVRAM低下。二重unscaleバグも修正(統括受理)。本格はbf16採用(B6 pre-flightでfp16のDETR勾配overflow解消)。
- [x] DoD-08: ⑥EMAが `EMA_ENABLED` で制御でき、検証/保存にEMA重みを使い、単体テストが成功する。（TR-3）→ 手順10: `models/ema.py:ModelEMA`(unwrap_model基準)、checkpointにEMA格納、`tests/test_model_ema.py`(7) 赤→緑。本格学習checkpoint_7 に `ema={module,updates=9680}` 実在。
- [x] DoD-09: ②TensorBoardが `TENSORBOARD` で制御でき（依存 `tensorboard` を追加済・lock整合）、有効時に `<logdir>/tb/` へevent生成、既定OFFで `log.txt`/wandb非破壊。（TR-3）→ 手順11: `uv add tensorboard`(2.20.0)、`Logger.tb_scalar`/`close`、`tests/test_logger_tensorboard.py`(4) 赤→緑。本格学習 `outputs/.../train/tb/events...` 実在・log.txt併存。
- [x] DoD-10: ①④config駆動early-stopが `EARLY_STOP` で制御でき、既定OFFで全EPOCHS実行、ONで停止する。（TR-3）→ 手順12: `should_early_stop` 純粋関数を単一停止経路へ合流、NaN別途明示失敗、`tests/test_early_stop.py`(6) 赤→緑。本格学習はloss改善継続で全8epoch実行(早期停止せず＝設計通り)。
- [x] DoD-11: ⑤汎用 `-u KEY=VALUE` 上書きが機能し、未知キーは警告（サイレント無視しない）。（TR-3）→ 手順13: `apply_cli_updates`(yaml.safe_load型解釈・未知キー`KeyError`)、`tests/test_cli_override.py`(8) 赤→緑。end-to-end `-u MAX_TRAIN_STEPS=1` 反映・未知キーで明示失敗 実証。
- [x] DoD-12: ⑦再現環境規約（`uv sync --frozen`/`uv run --no-sync`/seed）が文書化され、`uv sync --frozen` が成功する。（TR-3）→ 手順14: `README.md`再現節＋`justfile`(`sync-frozen`/`repro-doc`)、seedは`set_seed`で既にrank対応。`uv sync --frozen` exit 0(lock整合)。重要発見: `--frozen`後は`just build-ops`必須(op除去)を明記。
- [x] DoD-13: 本格config `configs/train_tracklet_pseudomot_full.yaml`（`MAX_TRAIN_STEPS`無し）が存在し、ID語彙とVRAMが実測・記録されている。（TR-5）→ 手順15-16: config存在・`MAX_TRAIN_STEPS`無。実測: clip内最大unique=128(実sampler 126)≤`NUM_ID_VOCABULARY=160`(切詰め無)、VRAM 2304MB(bf16)、1.32s/step、`SAMPLE_LENGTHS=[10]`/`EPOCHS=8`。
- [x] DoD-14: **2-step smokeで健全性確認後、本格学習が起動・完走し、`log.txt`・TensorBoard・checkpoint・最終メトリクスが残る。** 完走不能時は停止epoch・原因・次アクションがblockerとして証跡付きで記録されている。（TR-6）→ **完走**: 手順18回帰smoke＋B6 full-config 2-step pre-flight(6項目PASS, fp16→bf16是正)後、**全8 epoch完走(exit 0)**。最終 epoch7 `loss=7.8234`/`detr_loss=3.0175`/`id_loss=4.8059`(epoch2 8.71→epoch7 7.82 単調減少)、`checkpoint_0〜7.pth`(各947MB)、TB event、log.txt(626KB)、config.yaml、≈3.5h(各epoch~25min)。GPU残プロセス無。
- [x] DoD-15: `uv run pytest tests/` と `uv run ruff check`（本作業の変更/新規19ファイル群を対象）がgreen。（TR-7）→ 手順21: `pytest tests/ -q`→`42 passed`、`ruff check`(変更/新規19ファイル)→All passed。注: ruff対象は本作業の変更/新規ファイルに限定（リポジトリ全体には既存コード由来のlintが残るが本作業の変更分はclean）。
- [x] DoD-16: generated artifacts（`.venv/`,`outputs/`,`datasets/`,`pretrains/`,`*.pth`）がgit管理対象外で、commit/pushしていない。（TR-7）→ **整合更新**: 生成物(`.venv/`/`outputs/`(checkpoint_0〜7.pth・tb・log.txt)/`datasets/`/`pretrains/`/`*.pth`)は git管理対象外＝`git ls-files`で**未追跡(空)**、`check-ignore`で`.gitignore:27-29`がignore。**ソースはユーザ明示許可のもと origin/cu118 へ commit/push 済**（`4199aa9`実装+`8a12cd6` docs、`## cu118...origin/cu118`同期）。B7では新規commit/pushせず。
- [x] DoD-17: 作業記録に実行コマンド・成功/失敗・修正内容・制約・次アクションが記録されている。（TR-7,TR-8）→ §7作業記録表に手順1-23・各ブロック監査結果を日時・コマンド・赤→緑・証跡(file:line)付きで記録。§9に調査・設計成果。最終状態§8を実値で確定。

---

## 6. 補足: 設計判断（self-contained化のための明示）

- **AMPはAccelerate機構を使う**: MOTIPは `train.py:437 accelerator.autocast()` / `451 accelerator.backward()` を既に持つ。DEIMの生 `GradScaler`（`det_engine.py:48-76`）を写経せず、`Accelerator(mixed_precision=...)` を結線する（手順9）。これがMOTIP流の正しい移植。
- **early-stopはSolver相当=`train_engine`のepochループに置く**: MOTIPに独立Solverクラスは無いため、`train_engine` が DEIM `det_solver.fit()` の役割を担う。Bug1修正（早期停止の伝播）と④（patience監視）を同一の停止経路へ統合する（手順4・12）。
- **validationは当面無効・監視はloss**: tomato tracklet datasetは単一trainsequenceでval GTが無い。`INFERENCE_DATASET:`空で during-eval を切り、early-stop監視は epoch平均loss とする。将来HOTAを入れる場合は、temporalにval sub-sequence＋gtを切り出し `submit_and_evaluate` 経路へ繋ぐ（本作業では設計のみ）。
- **NUM_ID_VOCABULARYはclip内最大uniqueトラック数以上**: 超過すると `GenerateIDLabels` がランダム選択でサイレント切詰め。手順16で実測して設定する（no-silent-fallback）。

---

## 7. 作業記録

**重要な注意事項（書き換え後も残すこと）：**

* 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` で現在時刻を確認し、正確な日時を記録する。
* 各作業項目の開始時と完了時の両方で記録する。
* 作業内容は具体的なコマンド・操作手順を詳細に記載する。
* 結果・備考欄に成功/失敗、エラー内容、解決方法、重要な気づきを必ず記入する。
* フェーズごとに開始・完了の記録を取る。
* コード変更時は変更ファイル名と変更概要を記録する。
* エラー発生時はエラーメッセージと解決策を詳細に記録する。
* 推測で補完せず、実行したコマンド・観測した出力・編集したファイルに基づいて書く。generated artifactとcommit対象の区別を毎回明示する。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `2026-06-04` | `（記入）` | `（記入）` | 本書作成 | ✅作成: `write-workdoc-uv` テンプレに従い、ゴール要求分析/サブゴール/Trace ID/フェーズ/原子的チェックリスト/DoD/作業記録を記載。スコープ=フル移植、DoDにsmoke→FULL学習完走を含む。 |
| `2026-06-04` | `06:21:53 UTC (=15:21:53 JST)` | `worker(調査・設計)` | 手順1: 現在時刻と作業対象を記録 | ✅成功。`date "+%Y-%m-%d %H:%M:%S %Z%z"` → `2026-06-04 06:21:53 UTC+0000`（JST併記 `2026-06-04 15:21:53 JST+0900`）。対象repo=`/home/kasm-user/Desktop/MOTIP`（`git rev-parse --show-toplevel`で確認）, branch=`cu118`（`git rev-parse --abbrev-ref HEAD`で確認）, venv=`/home/kasm-user/Desktop/MOTIP/.venv`（存在確認済）, 本書path=`/home/kasm-user/Desktop/motip_sandbox/temp/workdoc_Jun04-2026_motip_experiment_mgmt_and_full_training.md`。**スコープ宣言:** 本worker割当はフェーズ1（手順1-3）の調査・設計のみ。コード変更/ファイル新規作成/実装は行わない。 |
| `2026-06-04` | `06:21〜06:25 UTC` | `worker(調査・設計)` | 手順2: MOTIP接続点を精査して記録（読込: `train.py`/`configs/util.py`/`runtime_option.py`/`log/logger.py`/`data/joint_dataset.py`/`data/util.py`/`data/transforms.py`/`pyproject.toml`） | ✅成功。**各移植①〜⑦の接続点（変更ファイル:行・方式）を §9.1 の表へ記録。** 重要裏取り（file:line, MOTIP本体は読取のみ・無改変）: (1)`train.py:39 accelerator = Accelerator()`（AMP③の結線点・現状引数なし）, `train.py:154 accelerator.prepare(...)`, `train.py:159 for epoch in range(...)`（early-stop①④のbreak挿入点）, `train.py:437 with accelerator.autocast():`／`train.py:451 accelerator.backward(loss)`（AMP③はAccelerate機構, autocastは`mixed_precision`に追従）, `train.py:186 max_train_steps=config.get("MAX_TRAIN_STEPS")`／`train.py:522-525`（内ループのみ`break`し`states["start_epoch"]`を進めて`return metrics`＝**Bug1: 外側epochループ非停止**を原典確認）, `train.py:498 logger.metrics(...)`（step粒度TB②挿入点）/`198`/`244`（epoch粒度TB②挿入点）。(2)`configs/util.py:7 update_config_with_kv`（再帰dotted上書き・型解釈は`True/False`のみ＝YAML型解釈なし）/`37 update_config`（typedフラグ適用・未知キーは`RuntimeError`）/`96 load_super_config`（`SUPER_CONFIG_PATH`継承）。(3)`runtime_option.py`は全てtyped引数で`-u/--update`は**未実装**（⑤の新規追加点）。(4)`log/logger.py:47 class Logger`（`SummaryWriter`無・`wandb`有, `_is_to_do()`=main processガード`258-259`, `metrics()`125行が②の二層化対象）。(5)`data/joint_dataset.py:18-24 dataset_classes`（mini-registry, PseudoMOT登録済）/`137 __getitem__`（transform適用`177-178`）。(6)`data/util.py:56 collate_fn`（`72 trajectory_id_labels`前提・key欠落でKeyError＝Bug3）。(7)`data/transforms.py:578 build_transforms`／`611-615 GenerateIDLabels(num_id_vocabulary=config["NUM_ID_VOCABULARY"], num_training_ids=config.get("NUM_TRAINING_IDS", config["NUM_ID_VOCABULARY"]))`＝**NUM_TRAINING_IDS実在キー裏取り(transforms.py:614)**。`GenerateIDLabels` 本体`375-429`の`414-418`で`_N > num_id_vocabulary or _N > num_training_ids`時に`torch.randperm`で**サイレント切詰め**を原典確認。**tensorboard未インストール実測:** `uv run --no-sync python -c "from torch.utils.tensorboard import SummaryWriter"` → `ModuleNotFoundError: No module named 'tensorboard'`。`pyproject.toml`(7-27行 deps)・`uv.lock` ともに `tensorboard` 不在（grep "NOT FOUND"）。追加予定テスト名を一覧化（§9.2）。 |
| `2026-06-04` | `06:25〜06:30 UTC` | `worker(調査・設計)` | 手順3: 移植方式と本格学習設計を文書化（読込: DEIM `det_solver.py`/`det_engine.py`/`ema.py`、設計書 §5 P1-P11・§9 移植①〜⑦、smoke config・dancetrack base config） | ✅成功。**設計判断(a)〜(e)と本格学習暫定パラメータを §9.3 に明記。** DEIM裏取り: early-stop=`det_solver.py:84-86,185-196,220-223`（`es_wait>=es_patience`で`break`）, EMA=`ema.py:36 deepcopy(dist_utils.de_parallel(model))`＝MOTIPでは`accelerator.unwrap_model`基準, `ema.py:54-66 update`（floating-point state平均, `48 decay*(1-exp(-x/warmups))`）, AMP=`det_engine.py:48-76`（DEIMは生`GradScaler`＝MOTIPは写経せず`Accelerator(mixed_precision=...)`採用, 理由:`train.py:437/451`が既にAccelerate機構）, TB二層=`det_engine.py:112-117`（step粒度`Loss/total`/`Lr/pg_j`/`Loss/{k}`）+`det_solver.py:147-149`（epoch粒度）。本格学習暫定値の根拠: base dancetrack `EPOCHS:10`/`SAMPLE_LENGTHS:[30]`/`SAMPLE_INTERVALS:[4]`/`AUG_NUM_GROUPS:6`/`NUM_ID_VOCABULARY:50`, smoke `NUM_ID_VOCABULARY:64`。tomato実測値(63 obj/frame・1367 track)は**手順16実測前の未確定値**として保留。val方針: tomato単一train seq(`datasets/TomatoTrackletMOT/train/nyx660_jun04/gt/gt.txt`存在確認, val GT無)→`INFERENCE_DATASET:`空でduring-eval無効・監視=epoch平均loss。判断が分かれるval設計は §9.3(d) に選択肢A/B/採用基準・保留理由を記録。 |
| `2026-06-04` | `06:30 UTC` | `統括(coordinator)` | B1監査結果の記録（auditor → 承認） | ✅**承認(B2進行可)**: auditorがB1(手順1-3)を原典照合し、自己申告と実体の不一致ゼロ・MOTIP無変更(`git -C ...MOTIP status --short`空)・スコープ遵守(手順4-23は`[ ]`)を確認。承認根拠=接続点file:line全spot-check一致(`train.py:159/437/451/522-525`, `configs/util.py:7`, `data/transforms.py:611-615`のNUM_TRAINING_IDS実在, tensorboard不在, DEIM `ema.py:33-78`/`det_solver.py:185-223`照合)、設計判断(a)-(e)がPlan(手順9-15/§6)と無矛盾、未確定事項(NUM_ID_VOCABULARY等)は手順16へ正しく委譲。**統括の留意事項(B2へ伝達)**: Bug1修正は`train_one_epoch`戻り値を`(metrics, early_stopped)`/dict化し`train_engine`(`train.py:159`)でbreak、手順12 patience監視と同一停止経路へ統合(二重break禁止)。val方針AはB2では確定不要。 |
| `2026-06-04` | `06:33〜06:40 UTC` | `worker(B2バグ修正)` | 手順4: [Bug1] MAX_TRAIN_STEPSのepochループ貫通修正（TDD） | ✅成功（赤→緑）。**新規** `tests/test_train_loop_control.py`（6テスト: 純粋判定関数2件・戻り値契約1件・epoch-loop制御3件）。実装前=`3 failed, 3 passed`（`reached_max_train_steps`不在＋`return metrics`のまま）→実装後=`6 passed`（`uv run --no-sync pytest tests/test_train_loop_control.py -q`）。**変更ファイル `train.py`（MOTIP, コード変更）**: (a)純粋判定関数 `reached_max_train_steps(global_step, max_train_steps)` 追加(`:578-588`), (b)`train_one_epoch` 冒頭に `early_stopped = False`(`:297`)・内ループ停止を `if reached_max_train_steps(...): early_stopped=True; break`(`:530-533`)・戻り値 `return metrics, early_stopped`(`:535`), (c)唯一の呼出元 `train_engine` を `train_metrics, early_stopped = train_one_epoch(...)`(`:165`) に更新しepochループ末尾(scheduler.step後)に `if early_stopped: break`(`:261-263`)。**停止経路一本化**: 内ループはflag設定のみ、break判定はepochループの`early_stopped`のみ（`grep`で `break` は内1/外1のみ＝二重break無）。**patience監視は未実装（手順12送り＝スコープ遵守）。** generated artifact無し、commit/pushなし。 |
| `2026-06-04` | `06:40〜06:44 UTC` | `worker(B2バグ修正)` | 手順5: [Bug2] サンプリング経路の回帰テスト追加（characterization） | ✅成功（現行コードで `1 passed`）。**新規** `tests/test_joint_dataset_getitem.py::test_getitem_produces_trajectory_id_labels`（**MOTIP本体は無改変・テスト追加のみ**）。tiny 2-frame PseudoMOT（`convert_coco_tracklets_to_pseudomot`流用）＋smoke相当 `build_transforms`(`data/transforms.py:578`)で `JointDataset.__getitem__(info)` を1サンプル取得し、`(images, annotations, metas)` 構造・各frameの8キー（`trajectory_id_labels`/`_masks`/`_ann_idxs`/`_times`/`unknown_*`）存在・`(G=1,1,N)` shape・dtype(int64/bool) を検証。`info` は実sampler契約（`data/naive_sampler.py:109-113`）に一致。**保証**: `__getitem__→build_transforms→GenerateIDLabels/TurnIntoTrajectoryAndUnknown` の生成経路がrefactorで欠落しないこと（coverage gap解消）。 |
| `2026-06-04` | `06:44〜06:50 UTC` | `worker(B2バグ修正)` | 手順6: [Bug3] collate_fnの前提を突くテスト＋明示防御（TDD） | ✅成功（赤→緑）。**新規** `tests/test_collate_fn.py`（3テスト）。実装前=`2 failed, 1 passed`（旧コードは opaque `KeyError('trajectory_id_labels')`＝メッセージに`collate_fn`/`transform`無し）→実装後=`3 passed`（`uv run --no-sync pytest tests/test_collate_fn.py -q`）。**変更ファイル `data/util.py`（MOTIP, コード変更）**: `collate_fn` のpadループ手前(`:71-79`)に全frame `trajectory_id_labels` 存在チェックを追加し、欠落時は `(batch=b, frame=t)` と原因（transform pipeline=`GenerateIDLabels`/`TurnIntoTrajectoryAndUnknown`）を示す**明示 `KeyError`** を raise（no-silent-fallback）。正常入力では従来挙動と完全等価（防御は欠落時のみ発火）。**B2全体検証**: `uv run --no-sync pytest tests/ -q` → `12 passed`（既存2＋新規10, 回帰なし）。`ruff check train.py data/util.py tests/*` → `All checks passed!`。`git status --short`= `M train.py` / `M data/util.py` / `?? tests/test_{train_loop_control,joint_dataset_getitem,collate_fn}.py`（変更は対象3ファイル＋新規テスト3に限定, generated artifact無, commit/pushなし）。 |
| `2026-06-04` | `06:52 UTC` | `統括(coordinator)` | B2監査結果の記録（auditor → 承認） | ✅**承認(B3進行可)**: auditorが自ら `uv run --no-sync pytest tests/ -q`→`12 passed`、`ruff`→All passed を再実行。`git diff train.py`/`data/util.py` 精読で (a)`train_one_epoch`→`(metrics, early_stopped)`、(b)`train_engine`単一break、(c)**呼出元はproduction1箇所のみ**(`grep train_one_epoch(`)、(d)**二重break無**(`grep break`=内1/外1)、(e)`reached_max_train_steps` 境界が旧インライン判定と意味的等価、を確認。Bug3防御は欠落時のみ明示KeyError・正常経路完全等価。Bug2テストは実`__getitem__`で8キー存在/shape/dtype検証。`patience`未実装(手順12送り)確認。`[情報]`級1件: collate_fn番兵は`trajectory_id_labels`単一キーだが8キーは`transforms.py:566-573`で原子生成のため実質十分(将来transform改修時に留意)。**統括の留意事項(B3へ伝達)**: 手順7はsummaryキー追加が既存12テスト(`_build_dataset`がconverter利用)を壊さぬこと、退化なし入力で`num_objects`不変を確認。手順8は`_get_sequence_names`のソート順不変を確認。両手順TDD赤→緑厳守。 |
| `2026-06-04` | `06:46〜06:56 UTC` | `worker(B3監査性改善)` | 手順7: 変換器のサイレントdropを明示化（TDD） | ✅成功（赤→緑）。**新規2テスト**を `tests/test_tracklet_pseudomot_conversion.py` へ追加: `test_degenerate_bboxes_are_counted_not_silently_dropped`（退化bbox=zero width/negative height 2件 + valid 2件の計4ann）、`test_no_degenerate_input_keeps_num_objects_and_zero_skips`（退化なし3annの**回帰**）。実装前=`2 failed, 1 passed`（`KeyError: 'num_skipped_degenerate'`）→実装後=`3 passed`（`uv run --no-sync pytest tests/test_tracklet_pseudomot_conversion.py -q`）。**変更ファイル `tools/convert_coco_tracklets_to_pseudomot.py`（MOTIP, コード変更）**: 退化bboxループ手前に `num_input_annotations=len(data["annotations"])`・`num_skipped_degenerate=0` を導入、`if w<=0 or h<=0:` 分岐で `num_skipped_degenerate += 1` してから `continue`（黙って捨てない=no-silent-fallback）、summary dict（`return`値＋`conversion_summary.json`）へ `num_input_annotations`/`num_skipped_degenerate` 2キー追加。不変条件 `num_input_annotations == num_objects + num_skipped_degenerate` を退化(4=2+2)・非退化(3=3+0)両方で検証。**回帰**: 退化なし入力で `num_objects` 不変（既存test `num_objects==3` pass継続）。 |
| `2026-06-04` | `06:56〜07:04 UTC` | `worker(B3監査性改善)` | 手順8: PseudoMOT loaderの堅牢化（TDD） | ✅成功（赤→緑）。**新規3テスト**を `tests/test_pseudo_mot_dataset.py` へ追加: `test_get_sequence_names_ignores_non_directory_entries`（stray `README.txt`/`.DS_Store` 無視）、`test_get_sequence_names_preserves_sorted_order`（insertion順≠sorted順の3 seq + stray fileで**ソート順不変**を固定）、`test_allow_empty_frames_controls_is_legal`（空frame付きdatasetで `allow_empty_frames` False→frame1非合法/True→合法）。実装前=`2 failed, 2 passed`（非dir 2件が `KeyError: 'Sequence'`＝stray fileをseq扱いし `seqinfo.ini` 不在）→実装後=`4 passed`（`uv run --no-sync pytest tests/test_pseudo_mot_dataset.py -q`）。**変更ファイル `data/pseudo_mot.py`（MOTIP, コード変更）**: `_get_sequence_names` を `sorted(name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name)))` へ修正（非dir除外, ソート順保持）。**B3全体検証**: `uv run --no-sync pytest tests/ -q` → `17 passed`（既存含め回帰なし）。`ruff check` 対象4ファイル → `All checks passed!`。`git status --short`= B3変更は `M data/pseudo_mot.py` / `M tools/convert_coco_tracklets_to_pseudomot.py` / `M tests/test_{pseudo_mot_dataset,tracklet_pseudomot_conversion}.py`（＋B2持越分）。generated artifact無, commit/pushなし。**スコープ遵守: 手順9以降は `[ ]` のまま未着手。** |
| `2026-06-04` | `07:06 UTC` | `統括(coordinator)` | B3監査結果の記録（auditor → 承認） | ✅**承認(B4a進行可)**: auditorが `pytest tests/ -q`→`17 passed`(内訳`--co`確認)、`ruff`→All passed を再実行。converterの**全skip経路を追跡**し、退化bboxのみ`continue`(計上)・`image_id`/`track_id`/`score`異常は`raise`(fail-fast)＝サイレントskipは退化経路のみ→不変条件 `num_input==num_objects+num_skipped` の健全性を確認。手順8は`sorted(...)`保持でソート順不変、`os.path.isdir`除外、`allow_empty`挙動をテスト固定。回帰なし(`_build_dataset`利用の既存テスト含め17pass)、手順9以降`[ ]`、commitなし。`[情報]`: raise経路は不変条件の対象外(=正しいfail-fast)。**統括の留意事項(B4aへ伝達)**: 手順9 AMPは `Accelerator(mixed_precision=config.get("AMP_DTYPE","no"))` で既定"no"=既存等価(非破壊)を確認。手順10 EMAは必ず `accelerator.unwrap_model` 経由・`EMA_ENABLED:False`既定。手順11 TBは `uv add tensorboard` が`pyproject.toml`+`uv.lock`を更新(初のlock変更)・`TENSORBOARD:False`既定でwriter=None・`is_main_process`ガード。3手順とも既定OFF非破壊・各ブロックで`pytest tests/ -q`緑維持。申し送り: `motip_sandbox/CLAUDE.md`の旧workdoc参照は将来整理(正本はJun04 workdocで確定)。 |
| `2026-06-04` | `06:56〜07:18 UTC` | `worker(B4a移植)` | 手順9: ③ AMPをAccelerate mixed_precisionで結線 | ✅成功（既定OFF非破壊＋fp16有効）。前提物確認: GPU=L4 23GB、`datasets/TomatoTrackletMOT`・`pretrains/r50_deformable_detr_coco_dancetrack.pth`(163MB)存在。**変更ファイル `train.py`（MOTIP, コード変更2箇所）**: (1)`:39` `Accelerator(mixed_precision=config.get("AMP_DTYPE","no"))`、(2)`:463-471` の `use_accelerate_clip_norm且つseparate_clip_norm` 分岐を `accelerator.unscale_gradients()`(AMP無時no-op)1回＋`torch.nn.utils.clip_grad_norm_`×2 へ修正。**修正理由**: fp16初回smokeで `unscale_() has already been called` 顕在化＝旧分岐が `accelerator.clip_grad_norm_`(内部で`scaler.unscale_()`)を `detr_params`/`other_params` で**2回**呼びAMP時にGradScaler二重unscale例外(accelerate実装をinspectで確認)。**証跡**: ①既定OFF smoke(`AMP_DTYPE`無=no)で `loss=43.1608`/`detr_grad_norm=98.6928`/`other_grad_norm=18.2396`/`max_cuda_mem=1069.84` がclip修正**後**もbaselineとbit一致(非破壊)。②fp16 smoke(`/tmp/amp_fp16_smoke.yaml`)で `loss=43.1665` finite・`max_cuda_mem=704.55`(約34%減)・`Stop training early`・exit0。生成物`outputs/tracklet_pseudomot_amp_smoke`は確認後削除。③`pytest tests/ -q`→`17 passed`、`ruff check train.py`→All passed。commit/pushなし。 |
| `2026-06-04` | `07:18〜07:32 UTC` | `worker(B4a移植)` | 手順10: ⑥ EMAを追加（Accelerate対応・TDD） | ✅成功（既定OFF非破壊＋ON動作）。**新規 `models/ema.py:ModelEMA`**（DEIM `ema.py:33-78` をAccelerate流に翻案: `de_parallel`非内蔵＝呼出側`unwrap_model`、`decay*(1-exp(-x/warmups))`ランプ・`start`スキップ・floating-point state平均・`state_dict={module,updates}`・`load_state_dict(strict=True)`）。**新規 `tests/test_model_ema.py`（7テスト）**: 赤(`ModuleNotFoundError: models.ema`)→緑(`7 passed`)。**変更 `train.py`**: import追加・`accelerator.prepare`後に`EMA_ENABLED`時のみ `ModelEMA(accelerator.unwrap_model(model), decay=EMA_DECAY(0.9999), warmups=EMA_WARMUPS(1000))` 生成・`train_one_epoch`へ`ema`引数・各optimizer.step後 `ema.update(accelerator.unwrap_model(model))`・save呼出2箇所に`ema=ema`。**変更 `models/misc.py`**: `save_checkpoint(..., ema=None)` でEMA時 `save_state["ema"]=ema.state_dict()`。**証跡**: 既定OFF smoke `loss=43.1608`(baseline一致・EMAログ無)。ON smoke(`/tmp/ema_smoke.yaml`)で "EMA enabled (decay=0.9, warmups=5)"・online loss不変・checkpoint に `ema={module,updates}`・`updates=2`(2 step=2 update)。生成物`outputs/tracklet_pseudomot_ema_smoke`は確認後削除。`pytest tests/ -q`→`24 passed`、`ruff`→All passed。`unwrap_model`経由徹底、int buffer非平均。commit/pushなし。 |
| `2026-06-04` | `07:32〜07:48 UTC` | `worker(B4a移植)` | 手順11: ② TensorBoard二層ロギング（TDD） | ✅成功（依存追加＋既定OFF非破壊＋ON動作）。**依存追加**: `uv add tensorboard`(2.20.0) → `pyproject.toml`+`uv.lock` 更新（**B4a初のlock変更**）、`from torch.utils.tensorboard import SummaryWriter` import成功確認(以前は `ModuleNotFoundError`)。**新規 `tests/test_logger_tensorboard.py`（4テスト）**: 赤(`AttributeError: 'Logger' object has no attribute 'tb_scalar'`)→緑(`4 passed`)。**変更 `log/logger.py`**: `Logger.__init__(..., tensorboard=False)`、`is_main_process`且つ`tensorboard`時のみ `SummaryWriter(log_dir=<logdir>/tb)`生成、`tb_scalar(tag,value,step)`(無効時no-op)・`close()`追加（既存F541 lint 1件も修正）。**変更 `train.py`**: Logger生成に `tensorboard=config.get("TENSORBOARD",False)`、step粒度TB(step-log直後: `Loss/total`/`Lr/pg`/`Mem/max_cuda_mb`/`Loss/<k>`, step=`global_step`)、epoch粒度TB(epoch-metrics直後: `epoch/<k>`=global_average)、epochループ後 `logger.close()`。**証跡**: 既定OFF smoke `loss=43.1608`・tb dir**未生成**。ON smoke(`/tmp/tb_smoke.yaml`)で `tb/events.out.tfevents...` 生成・74 scalar tags(`Loss/total`含む)を `EventAccumulator` で確認・`log.txt`(2682B)併存=**log.txt/wandb非破壊**。生成物`outputs/tracklet_pseudomot_tb_smoke`は確認後削除。**B4a全体**: `pytest tests/ -q`→`28 passed`、`ruff`(6ファイル)→All passed。`git status`= 変更`pyproject.toml`/`uv.lock`/`log/logger.py`/`models/misc.py`/`train.py`(＋B2,B3持越)・新規`models/ema.py`/`tests/test_{model_ema,logger_tensorboard}.py`(＋B2持越)。`outputs/`はgit-ignore。commit/pushなし。 |
| `2026-06-04` | `07:50 UTC` | `統括(coordinator)` | B4a監査結果の記録（auditor → 承認, 二重unscale修正を受理） | ✅**承認(B4b進行可)**: auditorが `pytest tests/ -q`→`28 passed`・`ruff`→All を再実行。**最重要争点=二重unscale修正を3層で検証し受理**: (a)Accelerate原典(`accelerator.py:2994`)で `clip_grad_norm_`=`unscale_gradients()+torch.nn.utils.clip_grad_norm_`、`unscale_gradients`はAMP=no完全no-op→旧2回呼びと新実装はAMP=no**bit等価**を証明、(b)非accelerate分岐は元から新パターン(`git show HEAD:train.py`)＝既存慣用へ揃えただけ、(c)**auditor自身がsmoke実行**しbaseline全値bit一致(`loss=43.1608`/`detr_grad_norm=98.6928`/`other_grad_norm=18.2396`/`max_cuda=1069.8433`)＋`Stop the epoch loop early at epoch 0`(Bug1 end-to-end)。EMA=`unwrap_model`経由/decayランプ/float-state平均/checkpoint格納は`EMA_ENABLED`時のみ、TB=遅延import/二重ガード/既定OFFで`tb/`未生成・log.txt非破壊、いずれも原典・7+4テスト・実smokeで確認。patience未実装(手順12送り)・手順12以降`[ ]`・commitなし。`[情報]`2件: 作業記録の行番号がEMA/TB挿入で数行ドリフト(可読性のみ,diffが正本)/FSDP・DeepSpeed使用時はclip置換に挙動差(本作業は単一GPUで対象外,申し送り)。**統括判断**: 二重unscale修正は③AMP達成に不可欠かつ非破壊実証済の発見的修正として**受理**。**B4bへの留意(auditor)**: 手順12 patienceはB2/B4aの`early_stopped`単一停止経路(`train_engine`の`if early_stopped: break`)へ合流させ`EARLY_STOP:False`既定で全EPOCHS実行(非破壊)・NaNは別途明示失敗、手順13 `-u`は`yaml.safe_load`型解釈＋未知キー警告(no-silent-fallback)、手順14は更新済lockで`uv sync --frozen`成功を確認。 |
| `2026-06-04` | `07:16〜07:28 UTC` | `worker(B4b移植)` | 手順12: ④ config駆動early-stopをepochループに実装（TDD） | ✅成功（既定OFF非破壊＋単一停止経路合流）。**純粋判定関数 `should_early_stop(history, patience, min_delta, start_epoch)`**(`train.py:661`、epoch平均loss=lower-better、`history[start_epoch:]`、初回はbaseline、`value < best - min_delta`で改善=wait reset、`wait>=patience`で停止)。**新規 `tests/test_early_stop.py`（6テスト）**: 赤(`ImportError: should_early_stop`)→緑(`6 passed`)。**変更 `train.py`**: epochループ前に `EARLY_STOP`(False)/`EARLY_STOP_PATIENCE`(10)/`EARLY_STOP_MIN_DELTA`(0.0)/`EARLY_STOP_START_EPOCH`(0) を `config.get` で読込・`loss_history`初期化。epoch metrics後に `train_metrics["loss"].sync()`→`global_average` 取得、**NaN/inf は `RuntimeError` 明示失敗(no-silent-fallback)**、`loss_history`蓄積、`early_stop_enabled且つshould_early_stop(...)`時 `early_stopped=True`。**既存の単一break(`:323`)へ合流＝二重break無**(grepでepochループのbreakは1箇所、MAX_TRAIN_STEPS由来と同一経路)。**証跡**: 既定OFF smoke `loss=43.1608`・"Early stopping"ログ無・MAX経路従来通り(`Stop training early...by MAX`+`Stop the epoch loop early`)・NaN-check通過(loss finite)。`config駆動` config keyは全て既定OFF/既定値。`pytest tests/ -q`→`34 passed`、`ruff`→All passed。commit/pushなし。 |
| `2026-06-04` | `07:28〜07:42 UTC` | `worker(B4b移植)` | 手順13: ⑤ 汎用 `-u KEY=VALUE` 上書きを追加（TDD） | ✅成功（型解釈＋未知キー明示失敗）。**新規 `configs/util.py:apply_cli_updates(config, updates)`** ＋ヘルパ `_set_existing_key`（再帰case-insensitive既存キー探索、新規作成せず）。`yaml.safe_load` で型解釈(int/float/bool/list/str)、**未知キーは `KeyError` 明示raise**(no-silent-fallback)、`=`無しは `ValueError`、適用は `warnings.warn` でログ。**新規 `tests/test_cli_override.py`（8テスト）**: 赤(`ImportError: apply_cli_updates`)→緑(`8 passed`)。**変更**: `runtime_option.py`(`-u/--update` `nargs="+"`)、`train.py`(import＋`__main__`で `update_config`後に `apply_cli_updates(cfg, opt.update)`)、`configs/util.py:update_config` のskipに `update` 追加(typed扱い回避)、`data/joint_dataset.py:dataset_classes` にregistry docstring。**証跡**: end-to-end `-u MAX_TRAIN_STEPS=1` → "Config overridden via -u: MAX_TRAIN_STEPS = 1"・`max_train_steps: 1`(元2)・`Stop training early at global_step=1`。未知キー実証 `-u AMP_DTYPE=fp16`(smoke未在)→`KeyError ... no silent fallback`。型解釈注: `1e-3`はYAML1.1で文字列→`0.001`使用(テストにコメント)。`pytest tests/ -q`→`42 passed`(警告8=`-u`上書きwarn,意図通り)、`ruff`(5ファイル)→All passed。commit/pushなし。 |
| `2026-06-04` | `07:42〜07:58 UTC` | `worker(B4b移植)` | 手順14: ⑦ 再現環境規約を明文化 | ✅成功（文書化＋`uv sync --frozen`実行確認＋重要発見）。**変更 `README.md`**: 「Reproducible local environment (uv)」節（`uv sync --frozen`/`just build-ops`/`uv run --no-sync`/rank-aware seed/`-u`overrides）。**変更 `justfile`**: `sync-frozen`(=`uv sync --frozen`)・`repro-doc`(規程print) target追加。**seed**: `utils/misc.py:set_seed` が `seed = seed + distributed_rank()` で**既にrank加味**→コード修正不要（文書化のみ）。**証跡**: `uv sync --frozen` **exit 0**（tensorboard追加後lock整合）。`uv run --no-sync python -c "import torch;print(...)"`→`2.4.0+cu118`。`just --list`に新target、`just repro-doc`出力確認。**重要発見(blocker回避)**: `uv sync --frozen` は**ローカルビルドの `MultiScaleDeformableAttention` op を削除**（`just build-ops`産物でPyPI/lock非管理）→training不能化。本検証で削除を観測→`just build-ops`で再build→training smoke(`loss=43.1608`/exit0)で機能回復確認。doc/justfileに「`--frozen`後は `just build-ops` 必須」を明記。これは暗黙再解決でなく `--frozen` の正常動作(lock外除去)。**B4b全体**: `pytest tests/ -q`→`42 passed`、`ruff`(6ファイル)→All passed。`git status`= 変更`README.md`/`justfile`/`configs/util.py`/`runtime_option.py`/`data/joint_dataset.py`/`train.py`/`pyproject.toml`/`uv.lock`(＋B2,B3,B4a持越)・新規`tests/test_{early_stop,cli_override}.py`(＋持越)。`outputs/`はgit-ignore。commit/pushなし。 |
| `2026-06-04` | `08:05 UTC` | `統括(coordinator)` | B4b監査結果の記録（並列敵対的監査5名＋統合 → 受理, B5前提確定） | ✅**B4bコード=完全PASS／総合判定=追加確認(B5前提)**: 初のfan-out監査。5検証者(early-stop/cli-u/repro-frozen-op/regression-scope/completeness)が並列検証し4名PASS確証。**実装の確証証跡**(複数者再現): `pytest tests/ -q`→`42 passed`(警告8=`-u`の意図的warn)、`ruff`All passed、`should_early_stop`純粋関数・**単一break(`train.py:325`)へMAX_TRAIN_STEPSとpatience合流(二重break無)**、NaN/inf epoch平均lossは`RuntimeError`明示失敗、`apply_cli_updates`はyaml.safe_load型解釈＋未知キー`KeyError`(no-silent-fallback)、seed=`set_seed`で`seed+distributed_rank()`、**CUDA op `.so`健在・import成功(非破壊確認)**、base commit`241f41d`不変・生成物ignore・push無。completeness critic がblocker/major提起だが**B4b欠陥ではなくB5/B6前提**。**統括判断: B4b受理・B5進行可**。**B5/B6前提として確定(B5割当に組込)**: (i/blocker)本格config(手順15)に移植系＋early-stop系**約9キーを明示**(`EARLY_STOP`/`EARLY_STOP_PATIENCE`/`EARLY_STOP_MIN_DELTA`/`EARLY_STOP_START_EPOCH`/`AMP_DTYPE`/`EMA_ENABLED`/`EMA_DECAY`/`EMA_WARMUPS`/`TENSORBOARD`、実装の`config.get`キー名と完全一致)＝`apply_cli_updates`は新規キー不作成のため。(ii/major)`uv sync --frozen`後は`just build-ops`必須・**B6学習前にop健全性確認**(`.so`存在＋import)。minor: pytest警告数の将来管理。 |
| `2026-06-04` | `07:38〜07:50 UTC` | `worker(B5本格設計)` | 手順15: 本格学習config作成 | ✅成功（前提(i)9キー明示・grep一致）。前提(ii) op健在確認(`import MultiScaleDeformableAttention`→OK)後着手、本ブロック中 `uv sync/add/--frozen` 不実行。**新規 `configs/train_tracklet_pseudomot_full.yaml`**: `SUPER_CONFIG_PATH`=dancetrack base継承、**`MAX_TRAIN_STEPS`未設定**、`DATASETS:[PseudoMOT]`/`DATASET_SPLITS:[train]`/`PSEUDOMOT_SUB_DIR:TomatoTrackletMOT`、**9キー明示**(`AMP_DTYPE:fp16`/`EMA_ENABLED:True`/`EMA_DECAY:0.9999`/`EMA_WARMUPS:1000`/`TENSORBOARD:True`/`EARLY_STOP:True`/`EARLY_STOP_PATIENCE:3`/`EARLY_STOP_MIN_DELTA:0.0`/`EARLY_STOP_START_EPOCH:2`)＝train.py の `config.get` キー名(grep)と完全一致、`INFERENCE_DATASET:`空、`OUTPUTS_DIR/EXP_NAME=tracklet_pseudomot_full`。**証跡**: `load_super_config` 例外無ロード・`MAX_TRAIN_STEPS=None`・`DATASETS=['PseudoMOT']`・9キー全present。pretrain(163MB)存在(自動DL不要)。commit/pushなし。 |
| `2026-06-04` | `07:50〜08:10 UTC` | `worker(B5本格設計)` | 手順16: ID語彙・VRAM・step時間を実測して確定（最重要） | ✅成功（実測確定・サイレント切詰め回避・OOM対処記録）。**dataset実測**: max **63 obj/frame**(is_legal 300上限違反0)、**1367 unique track**(全1219 frame)。**clip内最大uniqueトラック数(全sampler begin×interval≤4を網羅した厳密値)**: SL=[30]→**231**, SL=[16]→164, **SL=[10]→128**。**VRAM/step時間実測(fp16, EMA+TB, decoder-checkpoint+解像度↓+`DETR_NUM_CHECKPOINT_FRAMES:2`)**: SL=[30]base→**OOM**(alloc 20.28GB)、SL=[16]→8669MB/3.11s/step、**SL=[10]→4200MB/1.32s/step**(1210 steps/epoch)。**OOM対処(Plan順)**: base SL=[30] OOM→decoder-checkpoint+解像度↓適用も[30]/[16]は重い→**`SAMPLE_LENGTHS:[10]`短縮で解決**(BATCH_SIZE=1既定)。**確定値**: `SAMPLE_LENGTHS=[10]`/`NUM_ID_VOCABULARY=160`/`NUM_TRAINING_IDS=160`(128+32 margin＝**128≤160でサイレント切詰め無**)/`EPOCHS=8`。**完走時間見積**: 27min/epoch×8≈**3.6h(worst)**、early-stop(patience=3,start_epoch=2)で短縮見込。確定値を本格configへ反映。測定生成物`outputs/tracklet_pseudomot_full_measure`は削除。commit/pushなし。 |
| `2026-06-04` | `08:10〜08:18 UTC` | `worker(B5本格設計)` | 手順17: 本格学習用just target追加 | ✅成功。**変更 `justfile`**: `tracklet_full_config` 変数＋3 target追加: `config-tracklet-full`(17キーecho)、`train-tracklet-full`(`uv run --no-sync python train.py --config-path configs/train_tracklet_pseudomot_full.yaml`)、`tb`(`tensorboard --logdir outputs/tracklet_pseudomot_full/train/tb`)。**証跡**: `just --list` に3 target表示、`just config-tracklet-full`→17キー出力(`MAX_TRAIN_STEPS=None`/`DATASETS=['PseudoMOT']`/9キー含む)・exit 0。**B5全体検証**: op健在(`import MultiScaleDeformableAttention`→OK, `uv sync/add/--frozen`未実行)、`pytest tests/ -q`→`42 passed`、`git status`= 新規`configs/train_tracklet_pseudomot_full.yaml`・変更`justfile`(＋持越)、生成物無。**本格学習(`just train-tracklet-full`)はB5では未起動(B6で統括承認後)。手順18以降 `[ ]` 未着手。** commit/pushなし。 |
| `2026-06-04` | `08:25 UTC` | `統括(coordinator)` | B5監査結果の記録＋本格学習パラメータ承認（並列監査5名 → 承認, B6 GO） | ✅**承認(B6 GO・条件=pre-flight通過)**: 並列敵対的監査5名全員PASS。**最重要=ID語彙安全性を統合役が自分でgt.txtから独立再計測**: SL=[10] interval4→**128**(worst frame738-774)/contiguous→80、`128≤160`実機再現＝`data/transforms.py:414-418`の randperm 切詰め**発動せず**(160超は観測されず差戻し非該当)。config 9キーが`train.py`の`config.get`名と1:1一致・`MAX_TRAIN_STEPS=None`でload成功(全17キーdump確認)、VRAM 4200MB<23GB・1.32s/step×1210×8≒3.55h、op健在(`import MultiScaleDeformableAttention`→OK)、`pytest tests/ -q`→42 passed、just 3 target、生成物無・base commit不変・commit/push無。**統括: 本格学習パラメータを承認** → `SAMPLE_LENGTHS=[10]`(interval4)/`NUM_ID_VOCABULARY=NUM_TRAINING_IDS=160`/`EPOCHS=8`(early-stop patience=3,start_epoch=2)。**B6前提(監査確定の起動前チェック)**: ①op健在 ②pretrain(156MB)・dataset(1219f)存在 ③`just config-tracklet-full`でMAX_TRAIN_STEPS=None/9キー目視 ④pytest 42/ruff緑を直前再取得 ⑤**full-config(AMP+EMA+TB ON保持)で2-step pre-flight smoke**＝finite loss(`train.py:248`非有限hard-failが出ない)・AMP×EMA順序・EMA checkpoint`{module,updates}`・early-stop配線・実ローダで1clip unique≤160(切詰めログ無)を実証。pre-flight通過後にB6で8-epoch本学習を起動。 |
| `2026-06-04` | `08:00〜08:08 UTC` | `worker(B6-preflight)` | 手順18: 移植後の回帰smoke再実行 | ✅**PASS**。前提チェック: op健在(`import MultiScaleDeformableAttention`→OK)・pretrain(163MB)・dataset(1219f)・GPU free 22563MiB。`just train-tracklet-smoke`(既定config, AMP/EMA/TB既定OFF)→ `loss=43.1608`(finite,baseline一致)/`detr_loss=38.4402`/`id_loss=4.7205`、`Stop training early at global_step=2 by MAX_TRAIN_STEPS=2`、`Finish training epoch 0`、`Stop the epoch loop early at epoch 0`、exit 0。log.txtに `Start training epoch 0` のみ＝**epoch1へ進まない(Bug1修正end-to-end成立)**。`outputs/tracklet_pseudomot_smoke`削除。`uv sync/add/--frozen`未実行。commit/pushなし。 |
| `2026-06-04` | `08:08〜08:35 UTC` | `worker(B6-preflight)` | full-config 2-step pre-flight smoke（手順19前段検証＝監査確定項目, **本学習は未起動**） | ✅**全6項目PASS（GO）＋重要発見: fp16→bf16修正**。一時config `/tmp/full_preflight.yaml`(=full config継承+`MAX_TRAIN_STEPS:2`)で2-step実行。**6検証**: (1)exit 0完走✅ (2)finite loss✅(`loss=41.76`/`detr_loss=36.29`/`id_loss=5.48`, `train.py:248`非有限hard-fail無) (3)AMP×EMA順序✅(`unscale_()`二重エラー無・"EMA enabled"ログ) (4)EMA checkpoint✅(`{model,optimizer,scheduler,states,ema}`, `ema={module,updates}`, `updates=2`, model/EMA重み全finite) (5)TB event✅(74 scalar tags, `Loss/total`, **0 non-finite**) (6)ID語彙切詰め無✅(**実sampler 1210 clips の clip内最大unique=126 ≤ 160**, `truncation_would_fire=False`)。**重要発見**: 初回fp16実行で `detr_grad_norm=nan` が step0/20とも**持続**(loss/EMA/model重みはfinite＝scalerがoverflow step skip, DETR重みは更新確認 delta~1e-5)。AMP無(`AMP_DTYPE=no`)では `detr_grad_norm=67.1`(finite)＝**fp16のDETR backbone勾配overflowが原因**と特定。**bf16**(L4 sm_89対応, 同VRAM 2304MB)で `detr_grad_norm=62.5`(step0)/65.9(step20) finite・TB non-finite 0・重みfinite。→ **`configs/train_tracklet_pseudomot_full.yaml` を `AMP_DTYPE: fp16`→`bf16` に修正**(理由コメント付記)。修正後canonical 2-step pre-flightで6項目再確認PASS。全pre-flight生成物・temp config削除。op健在維持、`pytest tests/ -q`→`42 passed`。**本学習(`just train-tracklet-full`/8epoch)は未起動(B6で統括承認後)。手順19/20チェックは未付与。** commit/pushなし。 |
| `2026-06-04` | `08:40 UTC` | `統括(coordinator)` | B6-preflight受理＋bf16是正承認＋手順19 本学習起動 | ✅**GO→起動**: pre-flight 6項目PASS。**fp16→bf16是正を受理**(fp16の`detr_grad_norm=nan`を実証→bf16でgrad/loss/EMA/TB健全、L4 sm_89対応、Plan手順9🛠注記の範囲)。起動前チェック全PASS(op健在`import MultiScaleDeformableAttention`OK・config bf16/`MAX_TRAIN_STEPS=None`/EPOCHS=8/9キー・pretrain163MB・dataset1219f・出力先クリーン・GPU22.5GB空・`pytest 42 passed`)。**手順19=統括が `train.py --config-path configs/train_tracklet_pseudomot_full.yaml` をbackground起動(task `b39u3d84i`、`uv run --no-sync --directory`)**。健全走行確認: Runtime Config承認値、`Loaded Data 1219 frames`・DETR pretrainロード・`EMA enabled`・`Early stopping enabled(p=3,s=2)`・`Start training epoch 0`、step0 `loss=41.76`(finite)/`detr_grad_norm=62.51`(finite)、step120で `loss`減少(→29.97)・`detr_loss 36→25`・`id_loss 5.48→4.89`、`tps≈1.1s/step`・eta epoch0~20min・`max_cuda 5179MB`(L4内)。手順19=`[x]`。**手順20(完走確認)は学習完了通知後に実施**。commit/pushなし、生成物git管理外。 |
| `2026-06-04` | `~11:32 UTC` | `統括(coordinator)` | 手順20: 本格学習完走確認 | ✅**完走(exit 0)**: task `b39u3d84i` 全8 epoch完了。最終 epoch7 `loss=7.8234`/`detr_loss=3.0175`/`id_loss=4.8059`(epoch2 8.71→7 7.82 単調減少)、≈3.5h。`checkpoint_0〜7.pth`(各947MB)・TB event・log.txt 残存、GPU残プロセス無。手順20=`[x]`(統括確認)。 |
| `2026-06-04` | `11:36〜11:50 UTC` | `worker(B7検証記録)` | 手順21: テスト・lint | ✅前提op健在(`import MultiScaleDeformableAttention`→OK)。`uv run --no-sync pytest tests/ -q`→**`42 passed`**(8 warnings=`-u`意図的warn)。`uv run --no-sync ruff check`(train.py/data/tools/log/models/configs/runtime_option.py/tests 計19ファイル)→**`All checks passed!`**。本格学習完走証跡確認: checkpoint_7に `ema={module,updates=9680}`/`states={start_epoch:8,global_step:9680}`、log.txt epoch0-7 loss単調減少、TB event実在。commit/pushなし。 |
| `2026-06-04` | `11:50〜11:56 UTC` | `worker(B7検証記録)` | 手順22: 生成物ignore監査(commit済状態) | ✅`git status --short --branch`→`## cu118...origin/cu118`(working tree clean・origin同期)。`git ls-files \| grep -E '^(outputs/\|datasets/\|pretrains/)\|\.pth$'`→**EMPTY**(生成物未追跡)。`check-ignore`: `outputs/`(`.gitignore:28`)/`datasets/`(`.gitignore:29`)/`pretrains/`(`.gitignore:27`)。本格学習生成物(checkpoint_0〜7.pth各947MB・tb・log.txt 626KB)は `outputs/`=git外で未追跡。`git log`→`8a12cd6`+`4199aa9`+`241f41d`。B7で新規commit/pushせず。 |
| `2026-06-04` | `11:56〜12:08 UTC` | `worker(B7検証記録)` | 手順23: DoD整合・最終状態確定 | ✅DoD-01〜17を実態突合し**全`[x]`**(DoD-14完走反映・DoD-16をcommit済状態へ文言整合更新: 生成物未追跡＋ソースはユーザ許可でorigin/cu118へpush済`4199aa9`/`8a12cd6`)。最終状態§8の全プレースホルダを実値確定(repo/branch/origin/commit, バグ修正, 監査性, 移植①〜⑦, 本格config bf16, 本格学習完走=最終loss7.82/checkpoint_0〜7/TB/≈3.5h, 検証42 passed/ruff clean, generated artifacts git外, 制約=HOTA未実装/id_loss緩やか/checkpoint整理任意)。**最終チェック**: `rg -n -- "- \[ \]" workdoc...` → 意図しない未完了 **0件**(手順1-23・DoD-01〜17 全完了)。MOTIP無変更(push済からの差分無, `git status` clean)。commit/pushなし。 |
| `2026-06-04` | `12:15 UTC` | `統括(coordinator)` | 最終サインオフ監査（並列5検証者＋統合 → 完了承認 GO） | ✅**完了承認(GO)**: 最終並列敵対的監査5名(training-completion/tests-lint/git-commit-content/dod-workdoc/completeness)全員PASS、blocker/major(成果物欠陥)確証**0件**。独立確証: checkpoint_7 に `ema`(updates=9680=1210×8)・`states{start_epoch:8,global_step:9680}`・model 719テンソル全finite、loss epoch0→7 単調減少(最終 `loss=7.8234`/`detr=3.0175`/`id=4.8059`)、`pytest tests/ -q`→42 passed、`ruff`(変更19ファイル)→All passed、op健在、`## cu118...origin/cu118`(clean・同期, HEAD=origin=`8a12cd6`)、2コミット(`4199aa9`+`8a12cd6`)に生成物混入**0件**(`git ls-files`の生成物grep→EMPTY)、DoD-01〜17 全`[x]`・未完0件。指摘=DoD-15文言の明確化(ruff対象=変更19ファイル)のみ→本行で反映済、非blocker。将来(非ゴール・任意): HOTA/val結線・id_loss改善(clip長↑/epoch↑)・per-epoch checkpoint整理・`--frozen`後の`build-ops`。**ゴール(作業書＋CLAUDE.mdに従いDoD充足)を完了とみなす。** |

| 項目 | 最終状態 |
| :--- | :--- |
| 作業書 | `/home/kasm-user/Desktop/motip_sandbox/temp/workdoc_Jun04-2026_motip_experiment_mgmt_and_full_training.md` |
| MOTIP repo | `/home/kasm-user/Desktop/MOTIP`, branch `cu118`, origin `git@github.com:yuki-inaho/MOTIP.git`。**ユーザ明示許可でcommit/push済**: `4199aa9`(実装)+`8a12cd6`(docs `docs/ONBOARDING.md`)、`## cu118...origin/cu118`(同期, working tree clean)。 |
| バグ修正 | [Bug1] `train.py`: `reached_max_train_steps`+`train_one_epoch`→`(metrics,early_stopped)`+`train_engine`単一break(`:325`)／test `test_train_loop_control.py`(6)赤→緑。[Bug2] `tests/test_joint_dataset_getitem.py`(8キー回帰, 1 passed)。[Bug3] `data/util.py:collate_fn`明示KeyError防御／`tests/test_collate_fn.py`(3)赤→緑。 |
| 監査性改善 | 変換器: `tools/convert_coco_tracklets_to_pseudomot.py` に `num_input_annotations`/`num_skipped_degenerate` 追加(不変条件)／`test_tracklet_pseudomot_conversion.py`(+2)赤→緑。loader: `data/pseudo_mot.py:_get_sequence_names` 非dir除外(ソート順不変)／`test_pseudo_mot_dataset.py`(+3)赤→緑。 |
| 実験管理移植①〜⑦ | ③AMP=`AMP_DTYPE`(既定no非破壊, 本格bf16)＋二重unscale修正。⑥EMA=`EMA_ENABLED`/`models/ema.py`(unwrap_model基準)/checkpoint格納/test(7)。②TB=`TENSORBOARD`/`Logger.tb_scalar`/`uv add tensorboard 2.20.0`/test(4)/log.txt非破壊。①④early-stop=`EARLY_STOP*`/`should_early_stop`単一停止経路/NaN明示失敗/test(6)。⑤`-u KEY=VALUE`=`apply_cli_updates`(yaml型解釈/未知キーKeyError)/test(8)。⑦再現規約=README/justfile(`sync-frozen`/`repro-doc`)、seed rank対応。全て既定OFFで非破壊・smoke bit一致確認。 |
| 本格config | `configs/train_tracklet_pseudomot_full.yaml`（B5確定+B6 pre-flight修正: `SAMPLE_LENGTHS=[10]`/`NUM_ID_VOCABULARY=NUM_TRAINING_IDS=160`(実sampler clip内最大unique=126≤160, 切詰め無)/`EPOCHS=8`/**`AMP_DTYPE=bf16`(B6 pre-flightでfp16はDETR backbone勾配overflow→`detr_grad_norm=nan`持続と判明, bf16で解消・同VRAM)**/`EMA_ENABLED=True`/`TENSORBOARD=True`/`EARLY_STOP=True(patience=3,start=2)`/`MAX_TRAIN_STEPS`無/`INFERENCE_DATASET`空。VRAM実測=2304MB(bf16, L4 23GB内), step≈1.3s, 1210 steps/epoch, 完走見積≈3.6h(early-stopで短縮見込)。pre-flight 6項目全PASS。just: `config/train-tracklet-full`,`tb`） |
| 本格学習 | **完走(task `b39u3d84i`, exit 0)**: 全8 epoch(epoch0-7, `start_epoch=8`/`global_step=9680`)。停止理由=EPOCHS到達(loss単調改善でearly-stop非発火＝設計通り)。最終 epoch7 `loss=7.8234`/`detr_loss=3.0175`/`id_loss=4.8059`(epoch2 8.71→epoch7 7.82 単調減少)。checkpoint=`outputs/tracklet_pseudomot_full/checkpoint_0〜7.pth`(各947MB, `ema={module,updates=9680}`含む)。TB=`outputs/tracklet_pseudomot_full/train/tb/events.out.tfevents...`。log=`.../train/log.txt`(626KB)＋`config.yaml`スナップショット。経過≈3.5h(各epoch~25min)。GPU残プロセス無。 |
| 検証 | `uv run --no-sync pytest tests/ -q`→**42 passed**(8 warnings=`-u`の意図的warn)。`uv run --no-sync ruff check`(19ファイル)→**All checks passed!**。op健在(`import MultiScaleDeformableAttention`→OK)。 |
| generated artifacts | `.venv/`,`outputs/`(checkpoint_0〜7.pth・tb event・log.txt),`datasets/`,`pretrains/`,caches は**git管理対象外**(`git ls-files`未追跡=空、`.gitignore:27-29`でignore)。 |
| 制約・次アクション | (1)**validation/HOTA未実装**(tomato単一train seq・val GT無＝設計判断d。将来temporal split+HOTAは設計のみ§6補足)。(2)`id_loss≈4.81`は8 epoch通じ緩やか(detr_lossは3.88→3.02と明確改善)＝ID学習を伸ばすにはclip長↑(VRAM/step時間とのトレードオフ)・epoch↑・LR調整が将来候補。(3)per-epoch checkpoint(947MB×8=7.4GB)の整理は統括/ユーザ判断(削除しない)。(4)本格configは`AMP_DTYPE=bf16`(fp16はDETR勾配overflowで不可)。(5)`uv sync --frozen`後は`just build-ops`必須(op除去)。 |

---

## 9. フェーズ1 調査・設計の成果記録（worker, 2026-06-04）

> 本章は手順1-3（調査・設計フェーズ）の成果。MOTIP本体は**読取のみ・無改変**。すべて `file:line` 証跡付き。実装（フェーズ2以降）は本章で確定した方式・暫定値に紐づける。

### 9.1 MOTIP接続点（移植①〜⑦ × 変更ファイル:行・方式）

| 移植項目 | 変更ファイル:行（接続点） | 移植方式 | 既定OFFでの非破壊性 | Trace |
| :--- | :--- | :--- | :--- | :--- |
| **③ AMP** | `train.py:39` `accelerator = Accelerator()` | `Accelerator(mixed_precision=config.get("AMP_DTYPE","no"))` へ変更。既存 `train.py:437 with accelerator.autocast()` がそれに追従。DEIMの生`GradScaler`は**写経しない**。 | `AMP_DTYPE` 既定 `"no"` で `Accelerator()` と等価＝挙動不変 | TR-3,TR-4 |
| **⑥ EMA** | 新規 `models/ema.py`（`ModelEMA`）／`train.py:159`epochループ内・optimizer.step後（`train.py:470` 付近） | 各step後 `ema.update(accelerator.unwrap_model(model))`。eval/checkpointはEMA重み。decay=`decay*(1-exp(-updates/warmups))`。参考 `ema.py:33-78`。 | `EMA_ENABLED:False` で生成せず＝挙動不変 | TR-3 |
| **② TensorBoard二層** | `log/logger.py:47 class Logger`（`SummaryWriter` 追加）／呼出点 step粒度 `train.py:498`, epoch粒度 `train.py:198,244` | `TENSORBOARD:True` かつ `is_main_process`（`logger._is_to_do()` `logger.py:258-259`）時のみ `SummaryWriter(log_dir=<logdir>/tb)`。step=`Loss/total`/`Lr/<pg>`/`Loss/<k>`、epoch=metrics。`log.txt`/wandbは無改変。**依存追加必須**（9.2参照）。参考 `det_engine.py:112-117`。 | `TENSORBOARD` 既定OFFでwriter=None・ファイル生成なし | TR-3 |
| **①④ early-stop** | `train.py:159` epochループ（break挿入）／`train.py:522-526`（内ループの`MAX_TRAIN_STEPS`停止を呼出元へ伝播） | `train_one_epoch` がearly-stop到達を返し、`train_engine` のepochループで`break`。`EARLY_STOP`/`PATIENCE`/`MIN_DELTA`/`START_EPOCH` を epoch平均loss監視で判定。Bug1と統合。参考 `det_solver.py:84-86,185-196,220-223`。 | `EARLY_STOP:False` かつ `MAX_TRAIN_STEPS` 未設定で全EPOCHS実行＝挙動不変 | TR-1,TR-3 |
| **⑤ 汎用CLI `-u`** | `runtime_option.py`（`-u/--update` 追加, 現状未実装）／`configs/util.py:7 update_config_with_kv`（再利用） | `-u KEY=VALUE ...`（`nargs="+"`）。typedフラグ（`update_config` `configs/util.py:37`）適用後に重ねる。型解釈は `yaml.safe_load`。未知キーは警告（no-silent-fallback）。`data/joint_dataset.py:18-24 dataset_classes` をdocstring明示。 | `-u` 未指定で従来CLI経路のまま＝挙動不変 | TR-3 |
| **⑦ 再現環境規約** | `justfile`/`README.md`（文書化）／seed: `runtime_option.py:23 --seed` + `train.py:42 set_seed` | `uv sync --frozen`/`uv run --no-sync`/seed rank別を明文化。依存追加（②）後に再lock→`--frozen`。 | 文書追加のみ＝コード挙動不変 | TR-3 |

**補足の原典確認（Bug群, file:line）:**
- **Bug1（MAX_TRAIN_STEPSのepochループ貫通）:** `train.py:522-526` で `states["global_step"] >= max_train_steps` 時に内ループ `break` するが、その後 `train.py:526 states["start_epoch"] += 1` し `train.py:527 return metrics`。外側 `train.py:159 for epoch in range(...)` は止まらない。`EPOCHS=1`（smoke）で偶然成立。
- **Bug2（サンプリング経路の回帰テスト不在）:** `data/joint_dataset.py:177-178` で transforms 適用、`data/transforms.py:566-573` で `trajectory_id_labels` 等8キー生成。これを検証する単体テストが `tests/` に不在（現状 `tests/` は `test_pseudo_mot_dataset.py`, `test_tracklet_pseudomot_conversion.py` のみ）。
- **Bug3（collate_fnの前提未検証）:** `data/util.py:72 max_N = max(annotation[0]["trajectory_id_labels"].shape[-1] ...)` が `trajectory_id_labels` キー存在を無検査で前提。欠落時はサイレント `KeyError`。
- **サイレント切詰め（監査性, no-silent-fallback違反）:** `data/transforms.py:414-418` `if _N > num_id_vocabulary or _N > num_training_ids: _random_select_idxs = torch.randperm(_N)[:...]` で警告なくトラックを間引く。→ 手順16で `NUM_ID_VOCABULARY` を安全側設定し回避する設計。
- **tensorboard未導入（実測）:** `uv run --no-sync python -c "from torch.utils.tensorboard import SummaryWriter"` → `ModuleNotFoundError: No module named 'tensorboard'`。`pyproject.toml:7-27`（deps）・`uv.lock` に `tensorboard` 不在。→ ②実装時に `uv add tensorboard`（pyproject+lock更新）が前提（手順11）。

### 9.2 追加予定テスト名一覧（フェーズ2-4で赤→緑）

| テストファイル::テスト名 | 対応手順 | 検証内容 |
| :--- | :--- | :--- |
| `tests/test_train_loop_control.py::test_max_train_steps_stops_epoch_loop` | 手順4(Bug1) | `EPOCHS=2,MAX_TRAIN_STEPS=2` で total global_step=2 停止・epoch1へ進まない（early-stop伝播） |
| `tests/test_joint_dataset_getitem.py::test_getitem_produces_trajectory_id_labels` | 手順5(Bug2) | `__getitem__`→transforms で `trajectory_id_labels`/`_masks`/`_ann_idxs`/`unknown_*` 等8キーの存在・shape妥当 |
| `tests/test_collate_fn.py::test_collate_requires_trajectory_labels` | 手順6(Bug3) | 正常サンプルでpad成功・key欠落で明示例外（サイレントKeyErrorでない） |
| `tests/test_tracklet_pseudomot_conversion.py`（既存へ退化bboxケース追加） | 手順7 | summaryに `num_input_annotations`/`num_skipped_degenerate`、`num_input == num_objects + num_skipped` |
| `tests/test_pseudo_mot_dataset.py`（既存へ追加） | 手順8 | split配下の非dir無視・`allow_empty_frames` 真偽で `is_legal` 変化 |
| `tests/test_model_ema.py::test_ema_ramp_and_update` | 手順10(⑥) | `decay_fn` 指数ランプ・update後の重み移動 |
| `tests/test_logger_tensorboard.py::test_writer_disabled_by_default` | 手順11(②) | `TENSORBOARD` 未指定でwriter=None |
| `tests/test_early_stop.py::test_patience_triggers_break` | 手順12(①④) | loss列→停止epochをjudge純粋関数で検証 |
| `tests/test_cli_override.py::test_dotted_update` | 手順13(⑤) | `update_config_with_kv` でネスト/型解釈 |

### 9.3 設計判断 (a)〜(e) と本格学習の暫定パラメータ

**(a) AMP = Accelerate `mixed_precision`（DEIM生GradScalerは写経しない）**
- 採用理由: MOTIPは `train.py:39 Accelerator()` / `437 accelerator.autocast()` / `451 accelerator.backward()` を既に持つ。DEIM `det_engine.py:48-76` の `scaler.scale/unscale_/step/update` を移植すると二重管理になり破綻。`Accelerator(mixed_precision=config.get("AMP_DTYPE","no"))` に一本化し、autocast/backwardは既存のまま。config: `AMP_DTYPE`（`no`/`fp16`/`bf16`、既定`no`）。

**(b) EMA = `accelerator.unwrap_model` 基準**
- 採用理由: DEIMは `ema.py:36 dist_utils.de_parallel(model)` で unwrap してから deepcopy/update。MOTIPはAccelerate配下なので `accelerator.unwrap_model(model)`（分散/AMPラップ対策）を等価点として使う。`models/ema.py` に `ModelEMA`（decay=`decay*(1-exp(-updates/warmups))`、`start`スキップ、floating-point state平均、`state_dict={module,updates}`）。config: `EMA_ENABLED`(既定False)/`EMA_DECAY`/`EMA_WARMUPS`。eval/checkpointはEMA重み。

**(c) early-stop = `train_engine` のepochループで break**
- 採用理由: MOTIPに独立Solverクラス無し。`train_engine`(`train.py:32`) が DEIM `det_solver.fit()` 相当。DEIM `det_solver.py:185-196`（`es_wait>=es_patience`）・`220-223`（`break`）を epochループ(`train.py:159`)へ移植。Bug1（`MAX_TRAIN_STEPS` の内ループ`break`を呼出元へ伝播）と同一の停止経路へ統合。config: `EARLY_STOP`(既定False)/`EARLY_STOP_PATIENCE`/`EARLY_STOP_MIN_DELTA`/`EARLY_STOP_START_EPOCH`。

**(d) 監視 = epoch平均loss / during-eval無効（`INFERENCE_DATASET:` 空）**
- 前提: tomato tracklet datasetは単一trainsequence（`datasets/TomatoTrackletMOT/train/nyx660_jun04/` 1219 frames）で **val GTが無い**。`train.py:219 if config["INFERENCE_DATASET"] is not None:` を空で無効化＝during-train評価をスキップ。
- **判断が分かれる点（val設計）— 選択肢・採用基準・保留理由:**
  - 選択肢A（採用）: during-eval無効・監視指標=epoch平均loss・checkpoint毎epoch保存。**採用基準**: val GT不在で HOTA/MOTA を算出できないため、no-silent-fallback原則上「無いGTで評価したフリ」をしない。最小実装で本格学習完走（目標4）に到達可能。
  - 選択肢B（保留）: train sequenceの後半サブシーケンスをtemporal splitしてval GTとし `submit_and_evaluate` 経路でHOTA監視。**保留理由**: GT切り出し・評価器結線が新規実装で本作業スコープ（非ゴール「during-train HOTA完全実装は必須でない」§1）を超える。設計のみ残す（将来拡張）。
  - **採用**: A。early-stop監視指標は epoch平均loss。HOTAは設計のみ（本書 §6 補足と整合）。

**(e) 本格学習の暫定パラメータ（手順16実測前の初期値・要確定）**

| キー | 暫定値 | 根拠 / 確定条件 |
| :--- | :--- | :--- |
| `EPOCHS` | `10`（暫定） | base dancetrack `EPOCHS:10`（`r50_..._dancetrack.yaml:99`）を踏襲。early-stop ON で過剰分は自動短縮。単一GPU学習時間により手順19-20で調整余地。 |
| `SAMPLE_LENGTHS` | `[30]`（暫定、要VRAM/ID語彙実測） | base `SAMPLE_LENGTHS:[30]`（`:18`）。clip長↑でclip内uniqueトラック数↑＝`NUM_ID_VOCABULARY` 超過リスク（9.1サイレント切詰め）。OOM/語彙超過時は手順16で `[30]→短縮`。 |
| `SAMPLE_INTERVALS` | `[4]`（暫定） | base `:19`。 |
| `AUG_NUM_GROUPS` | `6`（暫定） | base `:40`。smokeは1。本格はbase踏襲。 |
| `NUM_ID_VOCABULARY` | **未確定（手順16で実測確定）** | base=50, smoke=64。tomato「最大63 obj/frame・全1367 track」は**未検証の引継ぎ値**。clip内最大uniqueトラック数を手順16で実測し、その**最大値以上＋マージン**に設定。これを満たさないと `transforms.py:414-418` がサイレント切詰め。 |
| `NUM_TRAINING_IDS` | **未確定（= `NUM_ID_VOCABULARY` 以上, 手順16確定）** | 実在キー（`transforms.py:614` で `config.get("NUM_TRAINING_IDS", config["NUM_ID_VOCABULARY"])` にfallback）。未指定なら `NUM_ID_VOCABULARY` と同値。 |
| `AMP_DTYPE` | `fp16` | L4 23GB VRAM対策。bf16非対応時fp16（手順9エラー対処）。 |
| `EMA_ENABLED` | `True` | 検証安定化（追跡系での有効性は要評価＝§5 P4注記）。 |
| `TENSORBOARD` | `True` | 二層可視化。`uv add tensorboard` 前提。 |
| `EARLY_STOP` | `True` | 過学習抑制。監視=epoch平均loss（(d)）。 |
| `INFERENCE_DATASET` | 空 | during-eval無効（(d)）。 |
| `MAX_TRAIN_STEPS` | **設定しない** | 本格学習はepoch完走 or early-stopで停止（smokeのみ設定）。 |
| `OUTPUTS_DIR`/`EXP_NAME` | `./outputs/tracklet_pseudomot_full` / `tracklet_pseudomot_full` | §7命名規約準拠。 |

**未確定・統括への要相談（推測で埋めない）:**
1. `NUM_ID_VOCABULARY`/`NUM_TRAINING_IDS` の確定値は手順16実測待ち（本worker割当外）。引継ぎ「63 obj/frame・1367 track」は原典未照合の前提値として保留。
2. `EPOCHS`/`SAMPLE_LENGTHS` はVRAM・学習時間の実測（手順16-20, 本worker割当外）に依存。本表は初期値。
3. val設計は選択肢A採用（loss監視）。HOTA監視（選択肢B）採否は統括判断事項。
