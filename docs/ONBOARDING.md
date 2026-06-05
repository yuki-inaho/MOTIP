# LLM オンボーディングサマリー — MOTIP tomato tracklet 疑似正解学習

> 本資料は、本プロジェクト（MOTIP を用いた tomato 多物体追跡の疑似正解学習）に新任 LLM エージェントが参加する際の初期資料です。リポジトリ内パスは MOTIP リポジトリルート相対、`(local)` 付きはこのマシンのローカル作業資料（リポジトリ非管理）です。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** MOTIP（Multiple Object Tracking as ID Prediction）による tomato（トマト）多物体追跡（MOT）の学習。semi-auto MOT で生成した tracklet を人手ラベルの代替＝**疑似正解（pseudo-label）**として用いる。
- **最終成果物:** tomato tracklet 疑似正解で学習した MOTIP 追跡モデル（checkpoint）と、再現可能な学習パイプライン（config / justfile / tests）＋ DEIM 相当の実験管理基盤（AMP / EMA / TensorBoard / config 駆動 early-stop / CLI `-u` 上書き）。
- **ビジネス背景・価値:** 人手アノテーションコストの削減。semi-auto annotation → 追跡学習 を新規 PC（uv / CUDA11.8 / NVIDIA L4 23GB）で再現可能に回せる状態にする。
- **現時点の進捗サマリ:**
  - 確定バグ修正（`MAX_TRAIN_STEPS` の epoch ループ貫通 / `__getitem__`→transforms 回帰テスト / `collate_fn` 前提防御）。
  - 監査性改善（converter のサイレント drop を件数記録へ / PseudoMOT loader 堅牢化）。
  - 実験管理フル移植①〜⑦（AMP=Accelerate `mixed_precision` / EMA=`unwrap_model` / TensorBoard 二層 / config 駆動 early-stop / 汎用 `-u KEY=VALUE` / uv 再現規約）。すべて **既定 OFF・非破壊**。
  - 本格学習 config 確定（`SAMPLE_LENGTHS=[10]` / `NUM_ID_VOCABULARY=NUM_TRAINING_IDS=160` / `EPOCHS=8` / `AMP_DTYPE=bf16`）。本格学習（8 epoch）を起動し、loss は `41.8 → 約8`（detr_loss `36→3`）へ収束。
  - 品質: `uv run --no-sync pytest tests/ -q` → 42 passed、`ruff check` → All passed。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ラインです。

- **暗黙 fallback 禁止 (no-silent-fallback):** 依存・入力・pretrain・GPU メモリ・ID 語彙の不足は黙って代替へ流れず、明示的に失敗（例外 / `sys.exit`）させ証跡を残す。
- **生成物を git に入れない:** `.venv/` / `outputs/` / `datasets/` / `pretrains/` / `*.pth` / TensorBoard event / caches。`.gitignore` で管理。
- **uv 環境前提:** `UV_PROJECT_ENVIRONMENT=<repo>/.venv`、実行は `uv run --no-sync`。`justfile` の target を優先。
- **MOTIP は HuggingFace Accelerate ベース:** AMP は **`Accelerator(mixed_precision=...)`** で結線する。DEIM の生 `GradScaler` を写経しない（`train.py` は既に `accelerator.autocast()` / `accelerator.backward()`）。
- **実験管理移植は既定 OFF・非破壊:** `AMP_DTYPE=no` / `EMA_ENABLED=False` / `TENSORBOARD=False` / `EARLY_STOP=False` の既定で、従来挙動と **bit 一致**（baseline `loss=43.1608`）。
- **ID 語彙のサイレント切詰め禁止:** `NUM_ID_VOCABULARY ≥ clip 内最大 unique トラック数`。下回ると `data/transforms.py` の `GenerateIDLabels` が randperm で無警告に切り詰める。SAMPLE_LENGTHS を変える際は必ず再計測（`SAMPLE_LENGTHS=[10]`,interval4 → 最大 128）。
- **L4 では fp16 不可:** DETR backbone 勾配が fp16 範囲を overflow し `detr_grad_norm=nan`。**`AMP_DTYPE=bf16`** を使う（L4 sm_89 はネイティブ対応、同 VRAM・同速度）。
- **`uv sync --frozen` 後は `just build-ops`:** ローカルビルドの CUDA op `MultiScaleDeformableAttention` は lock 非管理で `--frozen` により削除される。学習起動前に op 健在（`import MultiScaleDeformableAttention`）を確認。
- **t-wada TDD / トレーサビリティ:** 変更は赤→緑のテストで担保。作業記録に日時・証跡・変更ファイルを残す。
- **commit / push は明示許可時のみ。** 既定の作業ブランチは `cu118`（origin `git@github.com:yuki-inaho/MOTIP.git`）。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 作業計画書 兼 記録書（正本/Plan） | `(local)` `motip_sandbox/temp/workdoc_Jun04-2026_motip_experiment_mgmt_and_full_training.md` | フェーズ1-7・手順1-23・DoD・作業記録。**唯一の正本** |
| 設計書 | `(local)` `motip_sandbox/temp/design_Jun04-2026_training_experiment_management.md` | 学習実験管理のディレクトリ構成・設計パターン（11 パターン）・移植ロードマップ |
| 調査レポート | `(local)` `motip_sandbox/temp/report_Jun04-2026_deim_experiment_management.md` | DEIM 実験管理スタックの原典照合レポート（file:line 索引） |
| 本格学習 config | `configs/train_tracklet_pseudomot_full.yaml` | 本番学習設定（bf16/EMA/TB/early-stop, MAX_TRAIN_STEPS 無し） |
| smoke config | `configs/train_tracklet_pseudomot_smoke.yaml` | 2-step 健全性確認用 |
| 5FPS/後段tracker作業記録 | `reports/workdoc_Jun04-2026_motip_5fps_id_finetune.md` | 5FPS ID fine-tune、saturation判定、後段ByteTrack風再ID付け、DoD、作業証跡 |
| テスト資産 | `tests/` | 変換 / loader / collate / getitem / early-stop / cli-override / ema / tensorboard / train-loop-control |
| 既存ドキュメント | `docs/{GET_STARTED,INSTALL,DATASET,TUTORIAL,MODEL_ZOO}.md` | MOTIP 本家のセットアップ・データ・チュートリアル |
| エージェントチーム運用 | `(local)` `/workspace/CLAUDE.md`, `.claude/roles/{worker,audit}.txt` | 統括 / 作業 / 監査の 3 ロール運用、承認ループ |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- config 調整（既存キーの値変更、`-u KEY=VALUE` 上書き）。
- テスト追加・既存テスト保守（TDD 赤→緑）。
- 実験管理機能の拡張（評価 HOTA/MOTA の during-train 評価、追加ロギング等）。
- 本格学習の起動・監視・成果物確認（pre-flight → background 起動 → 完走確認）。
- 変換 / データ前処理ツールの改善。

### 任せないタスク（要・統括/ユーザ承認）
- MOTIP 本体アーキテクチャの大規模改造（registry 全面再設計等）。
- **commit / push**（明示許可があるときのみ）。
- 本格学習パラメータ（`EPOCHS` / `SAMPLE_LENGTHS` / `NUM_ID_VOCABULARY`）の独断変更（VRAM・ID 語彙・完走時間の実測と承認が前提）。
- 依存追加・`uv.lock` 変更（再現性に影響、承認の上で `--frozen` 運用）。
- 生成物（checkpoint/dataset/動画）の git 追加。

## 5. インタラクション方針
- **回答スタイル:** 日本語、見出し＋箇条書き、コードや事実は `file:line` で引用。
- **回答手順:** 前提確認 → 調査（原典/実コード）→ 設計判断 → TDD 実装 → 検証（pytest/ruff/smoke）→ 記録。
- **禁止事項・注意:** 推測で断定しない。未確認は「未確認」と明示。既定 OFF・非破壊を崩さない。
- **秘匿情報の扱い:** pretrain / dataset は外部 provenance（生成物として git 外）。連絡先以外の個人情報は扱わない。

## 6. 試行タスク（オンボーディング演習）
1. `just train-tracklet-smoke` を実行し、`loss=43.1608`（finite）と `Stop the epoch loop early at epoch 0`（`MAX_TRAIN_STEPS` の epoch ループ停止＝Bug1 修正）を確認する。
2. `-u KEY=VALUE` 上書きを試す（例 `... -u EPOCHS=3`）。**config に存在しないキーは `KeyError`（no-silent-fallback）** になることを `configs/util.py:apply_cli_updates` と `tests/test_cli_override.py` で確認する。
3. `configs/train_tracklet_pseudomot_full.yaml` の `NUM_ID_VOCABULARY=160` が、なぜ clip 内最大 unique トラック数（128）以上である必要があるかを `data/transforms.py` の `GenerateIDLabels` を読んで説明する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 作業書の「作業記録」表に日時（`date "+%Y-%m-%d %H:%M:%S %Z%z"`）・実施内容・結果・変更ファイル・証跡を追記。
- **TBD の扱い:** 「未確認」「保留（選択肢＋採用基準＋理由）」として明示。推測で埋めない。
- **レビュー/承認フロー:** 作業エージェント（worker）→ 監査エージェント（auditor, opus）→ 統括（coordinator）の承認ループ。必要に応じ並列敵対的監査（複数 verifier ＋ completeness critic）。
- **その他の運用ルール:** 既定 OFF・非破壊 / no-silent-fallback / 生成物 git 外 / DRY・KISS・SOLID。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:**
  - 本体: `MOTIP`（origin `git@github.com:yuki-inaho/MOTIP.git`, branch `cu118`）。
  - 疑似正解の生成元・実験管理の参照実装: `DEIM_sandbox`（`(local)` `/workspace/Project/DEIM_sandbox`）。
- **代表的なコマンド:**
  ```bash
  export UV_PROJECT_ENVIRONMENT=$PWD/.venv
  uv sync --frozen && just build-ops          # 環境同期（--frozen 後は op 再ビルド必須）
  uv run --no-sync pytest tests/ -q            # 全テスト
  uv run --no-sync ruff check <files>          # lint
  just build-tracklet-pseudomot                # COCO tracklet → PseudoMOT(MOTChallenge) 変換
  just build-tracklet-pseudomot-5fps           # 30FPS疑似正解をstride=6で5FPS相当PseudoMOTへ変換
  just loader-tracklet-5fps                    # 5FPS相当PseudoMOTのloader smoke
  just train-tracklet-smoke                    # 2-step smoke（既定 OFF）
  just config-tracklet-full                    # 本格 config 主要キー表示
  just train-tracklet-full                     # 本格学習（bf16/EMA/TB/early-stop）
  just config-tracklet-5fps                    # 5FPS ID fine-tune config主要キー表示
  just train-tracklet-5fps                     # 5FPS相当datasetでID強化fine-tune
  just tb                                       # TensorBoard
  just infer-tracklet-full [N]                  # 学習済モデルで追跡推論 → JSON(+MOTChallenge txt)。N=フレーム数(0=全)
  just infer-tracklet-5fps [N]                  # 5FPS fine-tuned checkpointで30FPS全frame列へ推論
  just visualize-tracklet-full [N]             # 推論JSONを元フレームへ描画 → 注釈フレーム＋mp4
  just video-tracklet-5fps 3 0                 # 5FPS fine-tuned推論JSONから3fps mp4のみ生成
  just compare-tracklet-5fps                   # 旧full runと5FPS fine-tuneのID proxy metrics比較
  just retrack-tracklet-5fps                    # 5FPS推論bbox列をByteTrack風に再ID付け → JSON/MOT/summary
  just video-retrack-tracklet-5fps 3 0          # 後段tracker結果から3fps mp4のみ生成
  just optuna-retrack-tracklet-5fps 80 240      # Optunaで長tracklet重視の後段tracker param探索
  just video-optuna-retrack-tracklet-5fps 3 0   # Optuna best retrack結果から3fps mp4のみ生成
  just build-tracklet-pseudomot-retrack-optuna  # Optuna best retrack JSONをPseudoMOT疑似正解へ変換
  just config-tracklet-retrack-optuna           # retrack-optuna疑似正解fine-tune config主要キー表示
  just loader-tracklet-retrack-optuna           # retrack-optuna PseudoMOTのloader smoke
  just train-tracklet-retrack-optuna            # checkpoint_39からretrack-optuna疑似正解で1epoch fine-tune
  just infer-tracklet-retrack-optuna-ft 0 fp32  # retrack-optuna fine-tuned checkpoint_40で30FPS全frame推論
  just video-tracklet-retrack-optuna-ft 3 0     # retrack-optuna fine-tuned raw推論から3fps mp4のみ生成
  just compare-tracklet-retrack-optuna-ft       # Optuna疑似正解とfine-tune後raw出力をID proxy比較
  just build-applemots-pseudomot                # /home/kasm-user/Desktop/APPLE_MOTS → datasets/AppleMOTSPseudoMOT
  just build-applemots-coco                     # AppleMOTS → COCO bbox(category_id=1, MOTIP/inspection向け)
  just build-applemots-coco-deim                # AppleMOTS → COCO bbox(category_id=0, DEIM num_classes=1向け)
  just loader-applemots                         # AppleMOTS変換datasetのPseudoMOT loader smoke
  just config-applemots-smoke                   # AppleMOTS 2-step smoke config主要キー表示
  just train-applemots-smoke                    # AppleMOTS PseudoMOTで2-step training smoke
  just transplant-applemots-bft-schedulefree    # 公式BFT tracking重み → AppleMOTS target checkpointへ移植
  just train-applemots-bft-sl8-pretrain         # AppleMOTS SL8/interval4, BFT移植 + ScheduleFreeで4epoch pretrain
  just transplant-applemots-to-tomato           # AppleMOTS pretrain tracking重み → tomato retrack-optuna targetへ移植
  just train-tracklet-applemots-transfer-smoke  # AppleMOTS転移checkpointからtomato 60step smoke fine-tune
  just infer-tracklet-applemots-transfer 0 fp32 # tomato全1219frame推論 → JSON/MOT txt
  just video-tracklet-applemots-transfer 3 0    # AppleMOTS転移後の推論JSONから3fps mp4のみ生成
  just compare-tracklet-applemots-transfer      # Optuna疑似正解とAppleMOTS転移後raw出力をID proxy比較
  ```
  - 推論/可視化は `tools/infer_tracklet.py`（`RuntimeTracker` を frame毎に回し `outputs/.../infer/tracks.json` と `tracks_mot.txt` を出力）/ `tools/visualize_tracks.py`（JSON+元画像→ bbox+ID 描画）。既定で `checkpoint_7.pth` の EMA 重みを使用。出力は `outputs/`（git外）。
  - 5FPS ID fine-tuneは `tools/convert_coco_tracklets_to_pseudomot.py --frame-stride 6` で `datasets/TomatoTrackletMOT_5fps`（git外）を作り、`configs/finetune_tracklet_pseudomot_5fps_id16.yaml` で学習する。既定推論checkpointはsaturation判定で採用した `checkpoint_39.pth`。旧新比較は `tools/compare_track_json.py` が `unique_track_ids / detections` と track length proxy を出す。
  - 5FPS fine-tune単体では推論IDが毎フレーム変わるため、`tools/retrack_detections.py` は `tracks.json` の `track_id` を使わず bbox/score/category を検出列として扱い、ByteTrack風のIoU+速度予測でIDを付け直す。採用設定は `track_thresh=0.80`, `new_track_thresh=0.95`, `match_thresh=0.10`, `max_age=60`。成果物は `outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/`（git外）で、`summary.json` は raw比 `unique_ids_per_detection: 1.0 -> 0.062644`, `track_length_max: 114`, `bottom_to_top_tracks: 101` を記録する。確認用動画は `outputs/tracklet_pseudomot_5fps_id16/retrack_bytetrack/tracks.mp4`（1219 frames, 3fps, 800x600）。
  - Optuna探索は `tools/tune_retrack_detections.py`（`optuna>=4.9.0`）で実行する。現時点のbestは `track_thresh=0.85`, `low_thresh=0.4`, `new_track_thresh=0.85`, `match_thresh=0.01`, `low_match_thresh=0.01`, `max_age=240`, `nms_iou=0.55`, `velocity_weight=0.0`, `velocity_momentum=0.65`。成果物は `outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/`（git外）で、`summary.json` は `unique_ids_per_detection=0.027223`, `track_length_mean=36.733978`, `track_length_max=151`, `num_tracks_ge_120=18`, `long_upward_tracks=196`, `bottom_to_top_tracks=67` を記録する。確認用動画は `outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/tracks.mp4`。
  - Optuna best retrackを疑似正解化する場合は `tools/convert_track_json_to_pseudomot.py` で `datasets/TomatoTrackletMOT_retrack_optuna`（git外）を作る。`configs/finetune_tracklet_pseudomot_retrack_optuna.yaml` は `SAMPLE_LENGTHS=[8]`, `NUM_ID_VOCABULARY=224`, `RESUME_MODEL=checkpoint_39.pth`, `EPOCHS=41`。`checkpoint_40.pth` まで1epoch fine-tuneした結果、raw MOTIP推論は `25,712 detections / 25,712 unique IDs / track_length_mean=1.0` で、Optuna疑似正解の長trackletを内部IDへ転写できなかった。現時点の実用出力は `retrack_optuna/tracks.json` と `retrack_optuna/tracks.mp4` を優先する。
  - AppleMOTSを試す場合は、raw zip `/home/kasm-user/Downloads/APPLE_MOTS.zip` を `/home/kasm-user/Desktop/APPLE_MOTS` に展開し、`tools/convert_apple_mots_to_pseudomot.py` で `datasets/AppleMOTSPseudoMOT`（git外）へ変換する。instance maskは16bit PNGで `category_id * 1000 + instance_id` のMOTSエンコード。変換後は train 6 sequences / 1147 frames / 62,899 objects、testing 6 sequences / 1051 frames / 46,068 objects。`configs/train_applemots_pseudomot_smoke.yaml` は `SAMPLE_LENGTHS=[2]`, `NUM_ID_VOCABULARY=256`, `MAX_TRAIN_STEPS=2` のsmoke用。
  - DEIM用AppleMOTS COCOは `tools/convert_apple_mots_to_coco.py --category-id 0` で `datasets/AppleMOTSCOCO_DEIM`（git外）に作る。DEIM `CocoDetection(remap_mscoco_category=False)` は `num_classes=1` の場合 `category_id=0` を要求するため、`category_id=1` の汎用COCOとは分ける。DEIM smoke configは `(local)` `/workspace/Project/DEIM_sandbox/configs/deim_dfine/deim_hgnetv2_m_coco_applemots_smoke.yml`、retry0結果は bbox AP=0.216 / AP50=0.553 / AR100=0.317。
  - AppleMOTS BFT pretrainは `configs/train_applemots_pseudomot_bft_schedulefree_sl8_pretrain.yaml`（SL8/interval4, short-edge 384, ScheduleFree, bf16, EMA, EarlyStop=True）を採用。`checkpoint_3.pth` は loss 8.0283 / detr_loss 5.5941 / id_loss 2.4342 で4epoch完走し、`outputs/applemots_pseudomot_bft_schedulefree_sl8_pretrain/checkpoint_3.pth`（git外）にある。
  - AppleMOTS pretrainをtomato retrack-optunaへ移植したcheckpointは `pretrains/motip_applemots_tracking_to_tomato_retrack_optuna_sl20.pth`（git外）。60step smoke後の推論成果物は `outputs/tracklet_pseudomot_retrack_optuna_applemots_bft_schedulefree_smoke/infer_30fps/`。`tracks.mp4` は1219 frames, 3fps, 136MB。比較では Optuna疑似正解 mean track length 36.73 / unique ratio 0.0272 に対し、AppleMOTS転移後raw MOTIPは mean track length 1.0 / unique ratio 1.0 で未改善。現時点の実用出力は引き続き `outputs/tracklet_pseudomot_5fps_id16/retrack_optuna/tracks.json` と `tracks.mp4` を優先する。
- **依存ライブラリ:** torch 2.4.0+cu118 / torchvision / accelerate / tensorboard / einops / opencv-python / optuna / pycocotools / numpy<2 / pytest / ruff（詳細は `pyproject.toml` と `uv.lock`）。CUDA op: `models/ops`（`just build-ops`）。
- **連絡先/責任者:** yoshikawa@inaho.co（yuki-inaho）。

> ※本資料は必要に応じて拡張・縮退して構いません。記入済みドキュメントはバージョン管理してください。
