# 学習プロジェクトの実験管理 — ディレクトリ構成・設計パターン設計書

**日付:** 2026-06-04
**種別:** 設計指針書（prescriptive design guide）
**目的:** 物体検出・追跡などの深層学習“学習”プロジェクトを、**再現可能・比較可能・拡張容易**に保つための「ディレクトリ構成」と「設計パターン」を、実証済みの参照実装（DEIM_sandbox）に紐付けて規定する。
**対象読者:** 本リポジトリ群（MOTIP / DEIM_sandbox / 後続の学習プロジェクト）で学習コードを設計・改修する開発者およびエージェント。
**参照実装の絶対パス基準:** 本書のすべての `参考コード` は以下を基準とする。
- DEIM本体: `/workspace/Project/DEIM_sandbox/DEIM/`
- sandboxラッパ: `/workspace/Project/DEIM_sandbox/`
- 比較対象MOTIP: `/home/kasm-user/Desktop/MOTIP/`

> 本書は **self-contained**（外部文書なしで読める）／**self-consistent**（用語・規約・例が相互に矛盾しない）／**well-defined**（各規約に検証可能な基準がある）を満たすことを目標とする。最終節 §10 の自己検査チェックリストで自己整合を担保する。関連レポート: `temp/report_Jun04-2026_deim_experiment_management.md`。

---

## 1. スコープと前提

- **対象:** 単一マシン〜小規模分散（DDP）での学習・評価・チェックポイント・実験比較。
- **非対象:** 大規模クラスタのジョブスケジューラ、MLOps配信、ハイパラ自動探索基盤（必要なら本設計の上に外付け）。
- **前提環境:** Python + PyTorch + `uv`（lockによる再現環境）。Linux/CUDA。

---

## 2. 用語定義（well-defined）

| 用語 | 定義 |
|---|---|
| **experiment（実験）** | 1つの設定（config）で定義される学習の論理単位。一意の `exp_name` を持つ。 |
| **run（実行）** | 1つの experiment の1回の実行。`output_dir/<exp_name>/<run_id>` に出力が閉じる。 |
| **config layer（設定レイヤ）** | `runtime / dataset / base(model・最適化・拡張) / experiment(最上位)` の4層。下→上の順に上書き合成。 |
| **registry（登録機構）** | 文字列 `type:` からクラスを生成する factory。YAMLだけで構成要素を差し替え可能にする。 |
| **Solver** | 実験ライフサイクル（初期化・epochループ・チェックポイント・best選択・早期終了）を持つ層。 |
| **Engine** | 「1イテレーション/1エポックの学習・評価」だけを担う純粋関数群（`train_one_epoch`/`evaluate`）。 |
| **stage** | 学習レジームの区切り（例: Stage1=拡張あり、Stage2=拡張オフ微調整）。境界は `stop_epoch`。 |
| **generated artifact** | 学習で生成される再生成可能な大容量物（`.venv/`, checkpoint, dataset, ログ）。git管理外。 |
| **no silent fallback** | 入力・依存・前提の欠落時に黙って代替へ流れず、**明示的に失敗**（例外/exit）させる原則。 |

---

## 3. 設計原則（7原則）

1. **Config as code + CLI override.** 設定はファイル合成で宣言、実験差分は新ファイルを作らず CLI のドット記法上書きで表現。**実行コマンド自体が再現記録**になる。
2. **Registry/Factory で構成要素を疎結合に.** model/optimizer/scheduler/dataset/loss を `type:` で差し替え可能にし、コード分岐を増やさない。
3. **Solver と Engine の分離.** ライフサイクル（状態を持つ）と1ステップ計算（状態を持たない純粋関数）を分け、テスト容易性とロギング注入点を確保。
4. **Reproducibility first.** 依存はlockで凍結＋index明示＋git pin、seedはrank別、データ順はepoch依存だが再現可能。
5. **No silent fallback.** 依存欠落→`ImportError`、入力欠落→`raise`、pretrain無し→自動DLせずパス・取得元を記録して失敗。
6. **Generated artifacts out of git.** `.venv/`・checkpoint・dataset・ログは `.gitignore`。コードと成果物を分離。
7. **Observability built-in.** step粒度の数値ログ（TensorBoard）と epoch粒度の構造化ログ（`log.txt`）を二層で常時出力。設定スナップショットを run に保存。

---

## 4. 推奨ディレクトリ構成

### 4.1 リポジトリ構成（コード）

```
<project>/
├── pyproject.toml / uv.lock / .python-version    # [P10] 再現環境
├── justfile                                       # 再現可能な実行エントリ（just train-... 等）
├── README.md / docs/                              # ONBOARDING（環境再構築・smoke手順・規約）
├── configs/                                        # [P1] 4層の設定
│   ├── runtime.yml                                #   共通既定(print_freq/checkpoint_freq/amp/ema/output_dir)
│   ├── dataset/<name>.yml                         #   データセット定義（パス・クラス数・transform方針）
│   ├── base/{model,optimizer,dataloader,loss}.yml #   再利用される基底
│   └── exp/<exp_name>.yml                         #   実験最上位（__include__でbaseを継承し差分のみ）
├── engine/                                         # 本体ライブラリ（importパッケージ）
│   ├── core/      {config.py, registry.py}        # [P1][P2] YAML合成 + registry/create
│   ├── solver/    {base_solver.py, det_solver.py, engine.py}  # [P2][P3][P8] ライフサイクル + train/eval
│   ├── optim/     {optim.py, ema.py, warmup.py, lr_scheduler.py}  # [P4][P6]
│   ├── models/    {...}                           # registry登録されたmodel/loss
│   ├── data/      {dataloader.py, dataset/, transforms/}  # [P9]
│   └── misc/      {dist_utils.py, logger.py, metrics.py} # [P7] seed/分散/ロギング
├── tools/                                          # 変換・前処理・pretrain抽出など（CLIスクリプト）
├── tests/                                          # 単体・回帰（変換, dataset, __getitem__→collate 経路）
└── train.py                                        # 薄いエントリ: config構築 → Solver選択 → fit()/val()
```

**規約:**
- `engine/` は**純import可能**（副作用なし）。`train.py` は薄く保つ（30〜60行）。
- 生成物（`outputs/`, `datasets/`, `pretrains/`, `.venv/`, `*.pth`, `*.mp4`）は **`.gitignore`**（[P6]）。

**参考（DEIM, 絶対パス）:** この構成は DEIM の実体に対応する。
- エントリ薄層 → `/workspace/Project/DEIM_sandbox/DEIM/train.py:28-53`
- `engine/core` → `/workspace/Project/DEIM_sandbox/DEIM/engine/core/{yaml_config.py,yaml_utils.py,workspace.py}`
- `engine/solver` → `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/{_solver.py,det_solver.py,det_engine.py}`
- 4層config → `/workspace/Project/DEIM_sandbox/configs/{runtime.yml,dataset,base,deim_*}`

### 4.2 1 run の出力構成（成果物）

```
outputs/<exp_name>/                # = output_dir
├── config_snapshot.yml            # 合成後の最終config（再現用）★推奨追加
├── log.txt                        # epoch毎の構造化JSON（train_*/test_*/epoch/n_parameters）
├── tb/                            # TensorBoard event（step粒度 loss/lr, epoch粒度 metric）
├── last.pth                       # 直近（Stage1のみ毎epoch）
├── best_stg1.pth / best_stg2.pth  # stageごとのベスト
├── checkpoint0012.pth ...         # checkpoint_freq毎
└── eval/{latest.pth, 050.pth,...} # 評価器の生スナップショット
```

**state_dict契約（well-defined）:** checkpoint は最低限 `{model, optimizer, lr_scheduler, ema, last_epoch, date}` を持つ。resume はこの全体、tuning は `model`（必要なら head 再マップ）のみをロードする。
**参考:** `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:125-183`（保存ロジック）, `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/_solver.py`（state_dict/resume/tuning）。

---

## 5. 設計パターンカタログ

各パターンは **問題 → 解決 → 最小契約 → 参考コード(絶対パス:行) → MOTIPへの適用** で記述する。

### P1. 階層YAML + registry + CLI ドット上書き（Hydra代替）
- **問題:** 実験ごとに設定ファイルが増殖し、差分が追えない。
- **解決:** `runtime→dataset→base→exp` を `__include__` で合成（後勝ち `merge_dict`）。構成要素は `type:` で registry から生成。実験差分は `-u model.backbone.depth=50 optimizer.lr=1e-4` のように CLI で与える。
- **最小契約:** `parse_cli(["a.b=1"]) -> {"a":{"b":1}}`（値はYAMLとして型解釈）。`create(cfg)` は `type:` を解決し未知typeで例外。
- **参考:** `/workspace/Project/DEIM_sandbox/DEIM/engine/core/yaml_utils.py`（`load_config`/`merge_dict`/`parse_cli`/`dictify`）, `/workspace/Project/DEIM_sandbox/DEIM/engine/core/workspace.py`（`register`/`extract_schema`/`create`）, `/workspace/Project/DEIM_sandbox/DEIM/train.py:37-41`。
- **MOTIP適用:** MOTIPは自前YAML（`/home/kasm-user/Desktop/MOTIP/configs/`, `utils/misc.py:yaml_to_dict`, `configs/util.py:load_super_config`）で `SUPER_CONFIG_PATH` 継承のみ。`-u` 相当のドット上書きと registry を足すと実験バリエーションが激減できる。

### P2. Solver（状態）/ Engine（純粋関数）分離
- **問題:** 学習ループにライフサイクル・ロギング・チェックポイントが混在し肥大化。
- **解決:** `Solver.fit()` が epochループ・best選択・保存・早期終了を持ち、`train_one_epoch()/evaluate()` は「1epochの計算と数値返却」だけを担う。ロギング/EMA/scaler は kwargs で注入。
- **最小契約:** `train_one_epoch(...) -> dict[str,float]`、`evaluate(...) -> (stats, evaluator)`。Engineは副作用最小（writer等は引数）。
- **参考:** `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:35-227`（fit）, `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:24-177`（train/eval）。
- **MOTIP適用:** MOTIPは `/home/kasm-user/Desktop/MOTIP/train.py` に `train_engine`/`train_one_epoch` が同居。Engine側を純粋化し、早期終了/保存をSolver層に寄せると、現在のバグ（後述ギャップ①）も自然に解消する。

### P3. チェックポイント戦略（last / per-stage best / periodic）
- **問題:** 「最新」「最良」「定期」を区別せず上書きし、復旧・比較ができない。
- **解決:** `last.pth`（再開用）/ `best_stg1.pth`・`best_stg2.pth`（stageごと最良）/ `checkpoint{e:04}.pth`（`checkpoint_freq`毎）を分離。bestは監視指標で選定。
- **最小契約:** 保存は `is_main_process` のみ。保存前後で optimizer mode を eval/train に切替（[P6]ScheduleFree対応）。
- **参考:** `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:30-33`（_save_state, mode切替）, `:125-183`（last/best/periodic, EMA refresh）, 既定 `checkpoint_freq=12` は `/workspace/Project/DEIM_sandbox/configs/runtime.yml:3`。
- **MOTIP適用:** MOTIPは単純保存。検証指標（HOTA/MOTA）に基づく best 保存を追加すると実用モデル選択が可能。

### P4. EMA（指数ランプ decay、検証はEMA重み）
- **問題:** 生重みは検証が不安定。
- **解決:** `ModelEMA` を全floating-point state_dict（param+buffer）に適用。decay は `decay*(1-exp(-updates/warmups))` の指数ランプ、`start` で初期スキップ。**検証・bestは EMA 重みで実施**。
- **最小契約:** `update(model)` を各step後に呼ぶ。`state_dict={module,updates}`、`load_state_dict(strict=True)`。
- **参考:** `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78`、検証でEMA採用 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:133-142`、既定 `/workspace/Project/DEIM_sandbox/configs/runtime.yml:16-20`（`use_ema=False`）。
- **MOTIP適用:** 追跡系へのEMA有効性は要評価だが、検証安定化策として導入余地あり。

### P5. AMP（forward FP16 / loss FP32 分離）
- **問題:** メモリ・速度律速、かつ損失計算のFP16は数値不安定。
- **解決:** forwardは `torch.autocast` 下、損失は `torch.autocast(enabled=False)`（FP32）で計算。`GradScaler` で scale→unscale→clip→step→update。
- **最小契約:** `use_amp` config で有効化（既定off）。clipは `scaler.unscale_` 後に実施。
- **参考:** `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:48-76`、scaler既定 `/workspace/Project/DEIM_sandbox/configs/runtime.yml:10-13`。
- **MOTIP適用:** L4等のVRAM制約下で有効。MOTIPは現状AMP無し。

### P6. LRスケジュール（iter単位 flat-cosine + warmup）と Optimizer モード
- **問題:** epoch単位stepはバッチ/データ量変更でLR軌跡が変わり再現性が落ちる。ScheduleFree系は train/eval切替が必須。
- **解決:** `FlatCosineLRScheduler`（warmup²→flat→cosine→no-aug）を**iter単位**step。標準schedulerならepoch単位＋`LinearWarmup`。ScheduleFree最適化器のため Solver が `optimizer.train()/eval()` を「メソッドがあれば呼ぶ」形で汎用吸収。
- **最小契約:** `self_lr_scheduler = (lrsheduler is not None)`。`_set_optimizer_mode(mode)` は callable のときのみ呼ぶ（標準optimizerでは no-op）。
- **参考:** scheduler構築 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:44-50`、iter step `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:95-99`、epoch step `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:119-121`、mode切替 `:24-33`、`/workspace/Project/DEIM_sandbox/DEIM/engine/optim/lr_scheduler.py`, `warmup.py`。
- **最適化器カタログ（registry差替）:** `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/optim.py:48-251`
  - `AutoMuonWithAuxAdam`（`:53-195`）= **ndim∈{2,4}の重み行列→Muon / それ以外→Adam** に自動振り分け（Muon: lr0.01/momentum0.95/ns_steps5、Adam: adam_lr0.00025）。依存欠落は `ImportError`（`:132-135`）。
  - `AdamWScheduleFreeOptimizer`（`:198-245`）= `__new__` で外部 `schedulefree.AdamWScheduleFree` を返す **registry adapter**。LR schedule不要だが train/eval切替必須 ⇒ [P6]の `_set_optimizer_mode` が必須。
- **MOTIP適用:** registryで `AdamW↔Muon↔ScheduleFree` を YAML差替できる構造は、最適化器比較実験の基盤として有用。

### P7. 二層ロギング + 設定スナップショット
- **問題:** テキストログだけでは学習曲線が追えず、設定が散逸する。
- **解決:** step粒度（10step毎に `Loss/total`,`Lr/pg_j`,`Loss/{k}`）を TensorBoard、epoch粒度（COCO/HOTA等の指標）を TensorBoard + `log.txt`(JSON行)。run開始時に最終configをTB textと `config_snapshot.yml` に保存。
- **参考:** step `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:112-117`、epoch指標 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:147-149`、log.txt `:205-207`、評価器dump `:210-218`。
- **MOTIP適用:** MOTIPは `/home/kasm-user/Desktop/MOTIP/log/logger.py` の `log.txt` のみ。`SummaryWriter` を1本足すだけで非破壊にstep可視化を追加できる（移植優先度1）。

### P8. config駆動の早期終了
- **問題:** 過学習・plateau後の無駄な学習継続。
- **解決:** 監視指標（検出=val mAP、追跡=HOTA/MOTA）に対し `early_stop / patience / min_delta / start_epoch` を config（CLI `-u`含む）で制御。`top1 > es_best+min_delta` で改善、`patience` 回連続未改善で break。`start_epoch` で初期ディップ無視。disable時は既存挙動不変。
- **参考:** `/workspace/Project/DEIM_sandbox/deim_early_stop.patch`、適用後 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:76-86,185-196,220-223`。
- **MOTIP適用:** 監視指標を差し替えれば非破壊で導入可能。

### P9. 拡張スケジュールを学習レジーム境界（stop_epoch）として共有
- **問題:** 強い拡張のまま終盤まで学習すると最終精度が頭打ち。
- **解決:** `stop_epoch` を境に Stage1（mosaic/mixup/multiscale）→ Stage2（拡張オフ・固定スケール）へ。**同じ境界をEMA再スタート・best分離（[P3][P4]）と共有**して、終盤を推論条件へ近づける。
- **参考:** 拡張ポリシー `/workspace/Project/DEIM_sandbox/DEIM/engine/data/transforms/container.py`、multiscale/collate `/workspace/Project/DEIM_sandbox/DEIM/engine/data/dataloader.py`、境界利用 `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:97-100,125,162,179-183`。
- **MOTIP適用:** MOTIPの augmentation も epoch依存の段階制御に寄せられる（要設計）。

### P10. 再現環境（uv.lock frozen + index explicit + git pin + seed rank別）
- **問題:** 数ヶ月後/別マシンで版が揺れ再現不能。
- **解決:** `uv.lock` を `--frozen` 同期、CUDA wheel index を `explicit=true` で固定、git依存はrev pin。`requires-python` を狭く固定。seed は `base+rank`、データ順は `sampler.set_epoch`。
- **参考:** `/workspace/Project/DEIM_sandbox/pyproject.toml:10-49`（torch2.8.0+cu128, `schedulefree==1.4.1`, muon git rev pin, `package=false`）、`/workspace/Project/DEIM_sandbox/scripts/uv-sync-desktop.sh`、seed系 `/workspace/Project/DEIM_sandbox/DEIM/engine/misc/dist_utils.py`（△未照合）。
- **MOTIP適用:** MOTIPは `/home/kasm-user/Desktop/MOTIP/pyproject.toml`+`uv.lock` で cu118 index固定済み。`--frozen`/`uv run --no-sync` 運用とseed rank別を規約化すると更に堅い。

### P11. No silent fallback（依存・入力の明示失敗）
- **問題:** 暗黙の代替動作が誤った成果物を生む（監査不能）。
- **解決:** 依存欠落→`ImportError`、入力/pretrain欠落→自動DLせず `raise`＋取得元記録、数値破綻→スナップショット保存後 `sys.exit(1)`。
- **参考:** optimizer依存 `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/optim.py:132-135,215-218`、NaN/非有限loss `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:52-62,104-108`。
- **MOTIP適用:** 変換器のサイレントなbbox欠落（`/home/kasm-user/Desktop/MOTIP/tools/convert_coco_tracklets_to_pseudomot.py:101-102`）を、件数記録付きの明示処理に変える（既存backlog）。

---

## 6. 実験運用レシピ（how-to）

| やりたいこと | 手順 |
|---|---|
| **新データセット追加** | `configs/dataset/<name>.yml` を作成（パス/クラス数/transform方針）。`exp` から `__include__`。datasetクラスをregistry登録。 |
| **新モデル/損失追加** | `engine/models/` に実装し `@register()`。`configs/base/model.yml` で `type:` 指定。 |
| **新実験** | `configs/exp/<exp_name>.yml` を `__include__` で baseから継承し**差分だけ**記述。微差は新ファイル不要、`-u` で上書き。 |
| **学習** | `just train-<exp>`（内部で `train.py -c configs/exp/<exp>.yml --output-dir outputs/<exp> --use-amp -u ...`）。 |
| **再開(resume)** | `-r outputs/<exp>/last.pth`（last_epochから、optimizer/scheduler/ema込み復元）。 |
| **転移(tuning)** | `-t <ckpt>`（重みのみ、epochリセット、必要ならhead再マップ）。resumeと排他。 |
| **smoke** | epoch=1・画像縮小・`max_train_steps`相当でループ健全性のみ確認（成果物としない）。 |
| **比較** | `outputs/*/tb` を TensorBoard で重畳、`log.txt` をJSONで集計。 |

---

## 7. 命名・出力規約（well-defined）

- `exp_name`: `<model>_<dataset>_<purpose>`（例 `dfine_tomato_ft`）。半角小文字+`_`。
- `output_dir`: `outputs/<exp_name>/`。同名再実行は**上書きせず** `outputs/<exp_name>/<YYYYMMDD-HHMM>/` を切る（run分離）。
- checkpoint名は §4.2 の固定語彙（`last/best_stg1/best_stg2/checkpoint{e:04}`）。独自名を増やさない。
- すべての run に `config_snapshot.yml` を必置（無い run は「再現不能」とみなす）。

---

## 8. アンチパターン（やってはいけない）

1. **実験ごとに巨大configを全文コピー**（差分が消える）→ `__include__`+`-u`。
2. **epoch境界とstep上限を混同**（partial epochで scheduler/保存が誤動作）→ 早期終了は Solver の epochループで明示break（MOTIP現状の不具合、ギャップ①）。
3. **best/last/periodic を同一ファイルに上書き** → 用途別に分離（[P3]）。
4. **依存・入力欠落時の暗黙fallback** → 明示失敗（[P11]）。
5. **生成物をgitに混入**（`.pth`/dataset/動画）→ `.gitignore`（[P6]）。
6. **train/eval計算にライフサイクルを混ぜる** → Solver/Engine分離（[P2]）。

---

## 9. MOTIPへの適用ギャップ → 移植ロードマップ（優先度順）

| 優先 | 項目 | 根拠（DEIM絶対パス）/ MOTIP対象 | 効果 |
|---|---|---|---|
| ① | **早期終了をSolver epochループでbreak**（MOTIPの `MAX_TRAIN_STEPS` は内ループのみbreakで外epochを止めない＝EPOCHS>1で破綻） | DEIM `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:185-223` / MOTIP `/home/kasm-user/Desktop/MOTIP/train.py` | 既存バグ修正 |
| ② | **TensorBoard二層ロギング** | DEIM `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:112-117` / MOTIP `/home/kasm-user/Desktop/MOTIP/log/logger.py` | step可視化, 非破壊 |
| ③ | **AMP(forward FP16/loss FP32)** | DEIM `.../det_engine.py:48-76` / MOTIP `train.py` | VRAM/速度 |
| ④ | **config駆動 early-stop（監視=HOTA/MOTA）** | DEIM `/workspace/Project/DEIM_sandbox/deim_early_stop.patch` | 無駄学習削減 |
| ⑤ | **CLI `-u` ドット上書き + registry** | DEIM `/workspace/Project/DEIM_sandbox/DEIM/engine/core/{yaml_utils.py,workspace.py}` / MOTIP `configs/util.py` | 実験効率 |
| ⑥ | **EMA + 検証はEMA重み** | DEIM `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py` | 検証安定（要評価） |
| ⑦ | **uv `--frozen` + seed rank別の規約化** | DEIM `/workspace/Project/DEIM_sandbox/pyproject.toml` / MOTIP `pyproject.toml` | 再現性 |

---

## 10. 自己検査チェックリスト（self-consistent性の担保）

設計を満たすかを run/PR ごとに確認する。すべて検証可能な基準を持つ（well-defined）。

- [ ] 設定は4層（runtime/dataset/base/exp）に分かれ、実験は `__include__`+差分のみで定義されている。
- [ ] 実験差分は CLI `-u` で表現でき、実行コマンドだけで run を再現できる。
- [ ] `train.py` は薄く（〜60行）、ライフサイクルは Solver、1 epoch計算は Engine にある。
- [ ] checkpoint は `last/best_stg*/periodic` に分離し、state_dict契約（model/optimizer/scheduler/ema/last_epoch/date）を満たす。
- [ ] 各 run に `config_snapshot.yml` がある。
- [ ] step粒度（TB）と epoch粒度（TB+log.txt）の二層ロギングが出ている。
- [ ] 早期終了/保存は **epochループで明示break/保存**しており、step上限と混同していない。
- [ ] 依存・入力・pretrain欠落は明示失敗する（暗黙fallbackなし）。
- [ ] 生成物（`.pth`/dataset/`.venv`/動画/ログ）は `.gitignore` 済み。
- [ ] 依存は lock凍結＋index明示＋git pin、seedは rank別。

---

## 11. 参考コード 絶対パス索引（対応箇所）

| トピック | 絶対パス:行 |
|---|---|
| エントリ（config構築/solver選択） | `/workspace/Project/DEIM_sandbox/DEIM/train.py:28-53` |
| YAML合成/CLIパース | `/workspace/Project/DEIM_sandbox/DEIM/engine/core/yaml_utils.py` |
| registry/create | `/workspace/Project/DEIM_sandbox/DEIM/engine/core/workspace.py` |
| Solver: fit/best/EMA refresh/保存 | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:35-227`（保存 `:125-183`, mode切替 `:24-33`, 早期終了 `:76-86,185-196,220-223`） |
| Engine: train/eval/AMP/NaN/TB/scheduler | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:24-177`（AMP `:48-76`, NaN `:52-62,104-108`, TB `:112-117`, iter-step `:95-99`） |
| Solver基底: setup/resume/tuning/state_dict | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/_solver.py` |
| EMA | `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78` |
| 最適化器(Muon/ScheduleFree/標準) | `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/optim.py:48-251` |
| LRスケジューラ/warmup | `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/{lr_scheduler.py,warmup.py}` |
| DEIM損失/マッチング(MAL等) | `/workspace/Project/DEIM_sandbox/DEIM/engine/deim/{deim_criterion.py,matcher.py}` |
| データ/拡張スケジュール | `/workspace/Project/DEIM_sandbox/DEIM/engine/data/{dataloader.py,transforms/container.py,transforms/mosaic.py,dataset/_dataset.py}` |
| seed/分散 | `/workspace/Project/DEIM_sandbox/DEIM/engine/misc/dist_utils.py`（△未照合） |
| 共通既定(runtime) | `/workspace/Project/DEIM_sandbox/configs/runtime.yml` |
| base(model/最適化/拡張/損失) | `/workspace/Project/DEIM_sandbox/configs/base/deim.yml` |
| 再現環境 | `/workspace/Project/DEIM_sandbox/pyproject.toml` / `/workspace/Project/DEIM_sandbox/scripts/uv-sync-desktop.sh` |
| 独自patch | `/workspace/Project/DEIM_sandbox/{deim_early_stop.patch,data_patch.patch,new_timm_model.patch}` |
| オンボーディング | `/workspace/Project/DEIM_sandbox/docs/{ONBOARDING.md,LLM_ONBOARDING_SUMMARY.md}` |
| 比較対象 MOTIP | `/home/kasm-user/Desktop/MOTIP/{train.py,log/logger.py,configs/,pyproject.toml,tools/convert_coco_tracklets_to_pseudomot.py}` |
