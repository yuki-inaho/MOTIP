# DEIM_sandbox 実装構成 と 学習の実験管理 — 調査レポート（原典照合済み）

**日付:** 2026-06-04
**対象:** `/workspace/Project/DEIM_sandbox`（本体は git submodule `DEIM/`）
**作成方法:** 5観点の並列サブエージェント調査 → 統合 → コーディネータが主要主張を `file:line` で原典再確認し、矛盾・誤りを訂正。
**引用表記:** 本文中は可読性のため相対表記を使うが、**絶対パス基準**は次の通り。
- `DEIM/<...>` ＝ `/workspace/Project/DEIM_sandbox/DEIM/<...>`（例 `engine/solver/det_solver.py:97` → `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:97`）
- `DEIM_sandbox/<...>` ＝ `/workspace/Project/DEIM_sandbox/<...>`
- 比較対象 MOTIP ＝ `/home/kasm-user/Desktop/MOTIP/<...>`
- 主要ファイルの絶対パス一覧は最終節「§8 参考コード 絶対パス索引」を参照。

> 本レポートの「✅確認済」は本セッションでコーディネータが実ファイルを読んで確認した事実。「△要確認」は未照合で残った点。

---

## 0. 全体像

### 0.1 リポジトリ構成

```
DEIM_sandbox/
├── pyproject.toml / uv.lock / .python-version   # 再現環境: uv, py3.10, torch2.8.0+cu128
├── deim_early_stop.patch                        # 独自: config駆動の早期終了 (val mAP監視)
├── data_patch.patch                             # 独自: num_workers 8→4
├── new_timm_model.patch                         # 独自: timm層名 dot→underscore マッピング修正
├── configs/                                     # sandbox側の実験設定 (DEIM/configs を __include__ で継承・上書き)
│   ├── runtime.yml                              # 共通既定 (print_freq/checkpoint_freq/amp/ema)
│   ├── dataset/*.yml  base/*.yml  deim_*/*.yml
├── docs/{ONBOARDING.md, LLM_ONBOARDING_SUMMARY.md}
├── scripts/uv-sync-desktop.sh
├── export_onnx.py / onnx_batch_infer.py
└── DEIM/  (submodule = 本体)
    ├── train.py                                 # エントリポイント
    └── engine/
        ├── core/   {yaml_config.py, yaml_utils.py, workspace.py}  # 設定システム(registry)
        ├── solver/ {_solver.py, det_solver.py, det_engine.py}     # 実験ライフサイクル + 学習/評価ループ
        ├── optim/  {optim.py, ema.py, warmup.py, lr_scheduler.py} # 最適化/EMA/スケジューラ
        ├── deim/   {deim_criterion.py, matcher.py}                # DEIM固有損失(MAL等)
        └── data/   {dataloader.py, dataset/, transforms/}         # データ/拡張パイプライン
```

### 0.2 実験管理スタック要約

| 観点 | 実現手段 | 該当 file:line | 既定 |
|---|---|---|---|
| 設定管理 | 階層YAML `__include__` + `merge_dict` + registry(`type:`) + CLI `-u a.b=val`(dot記法) | `engine/core/yaml_utils.py`, `workspace.py`, `train.py:37-41` | — |
| ロギング | TensorBoard(step粒度) + `log.txt`(epoch JSON) + config全体をTB textに保存 | `engine/solver/det_engine.py:112-117`, `det_solver.py:147-149,205-207` | — |
| チェックポイント | `last.pth`(stage1のみ毎epoch) / `best_stg1.pth` / `best_stg2.pth` / `checkpoint{e:04}.pth`(freq毎) | `det_solver.py:125-183` | `checkpoint_freq=12` |
| EMA | `ModelEMA` 指数ランプ decay、検証はEMA重みで実施、stage遷移でdecayリセット | `engine/optim/ema.py:33-66`, `det_solver.py:97-100,179-183` | `use_ema=False`, decay0.9999/warmups1000 |
| AMP | `autocast`(forward FP16) + `autocast(enabled=False)`(loss FP32) + `GradScaler` | `det_engine.py:48-76` | `use_amp=False` |
| スケジューラ | `FlatCosineLRScheduler`(warmup²→flat→cosine→no-aug)をiter単位step、または epoch標準scheduler+`LinearWarmup` | `engine/optim/lr_scheduler.py`, `warmup.py`, `det_solver.py:44-50,119-121` | — |
| 早期終了 | 独自patch: val mAP(top1)監視, `early_stop*` config駆動 | `det_solver.py:76-86,185-196,220-223` | `early_stop=False` |
| 最適化器 | AdamW/SGD/Adam + 独自 `AutoMuonWithAuxAdam` + `AdamWScheduleFreeOptimizer`(registry adapter) | `engine/optim/optim.py:48-251` | — |
| 再現環境 | `uv.lock` frozen + index `explicit=true`(cu128) + muon git pin + seed(rank別) | `pyproject.toml`, `scripts/uv-sync-desktop.sh` | py3.10/torch2.8.0 |

---

## 1. 設定・実験定義システム（engine/core, configs）

**何をしているか** — Hydra等の外部FWを使わず、軽量自前YAML合成系を持つ。

- **階層継承**: YAMLの `__include__` で base→dataset→model→実験設定を再帰合成。`load_config()` が深さ優先で処理、`merge_dict()` で後勝ち上書き（`engine/core/yaml_utils.py`）。最後に評価される実験ファイルが最優先。
- **CLI上書き**: `train.py:37-41` で `parse_cli(args.update)` がCLI `-u a.b=1 c=2` を `{'a':{'b':1},'c':2}` に変換（`yaml.load` で値型解釈）、`merge_dict` でYAMLへ上書き。さらに `args.__dict__`（`--seed/--use-amp/--output-dir/--summary-dir` 等の非None）もmerge。
- **遅延生成(registry/factory)**: `@register()`（`workspace.py`）がクラスの `__init__` シグネチャから schema 抽出、`create()` が YAML の `type:` から依存を再帰解決し `__inject__`/`__share__` を注入。`YAMLConfig` の各構成要素（model/optimizer/...）は `@property` で遅延評価。
- **task→solver**: `TASKS[cfg.yaml_cfg['task']](cfg)`（`engine/solver/__init__.py`: `detection→DetSolver` 等）。

**工夫** — ①新configファイルを作らずネスト深部まで `-u` で上書きでき、**実行コマンド自体が再現記録**になる。②遅延生成で `--test-only` 時に不要モジュールを作らない。③schema自動抽出で手動型登録不要。

**弱点** — sweep(grid search)機構は無し（外部bashループ依存）。config validationは最小で、不正 `type:`/`__inject__` は実行時例外。output_dirへのconfig自動コピーは無し（ただしTB textには保存: `det_solver.py` writer.add_text 系）。

---

## 2. 学習ライフサイクル・チェックポイント（engine/solver）✅原典確認済

`det_solver.py:35-227 fit()` を実コードで精査。要点（✅は確認済）:

- **エントリ/排他**: `train.py` で `-r/--resume`（last_epochから継続）と `-t/--tuning`（重みのみロード, epochリセット）は `assert not all([tuning, resume])` で排他。
- **初期化順序**（`_solver.py`）: device → tuningロード(**EMA生成前**) → DDPラップ → criterion/postprocessor → **EMA/Scaler生成** → writer。この順序は EMA が正しい初期重みを deepcopy するために重要（`ema.py:30-31` のコメント参照）。
- **scheduler選択** ✅: `args.lrsheduler is not None` のとき `FlatCosineLRScheduler` を構築し `self_lr_scheduler=True`（`det_solver.py:44-50`）。この場合は**iter単位**で `lr_scheduler.step(cur_iters+i, optimizer)`（`det_engine.py:95-96`）。そうでなければ epoch単位 `lr_scheduler.step()` を **warmup完了後のみ**（`det_solver.py:119-121`）+ iter単位 `lr_warmup_scheduler.step()`（`det_engine.py:98-99`）。
- **2-stage学習** ✅: `collate_fn.stop_epoch` を境界に Stage1（多スケール+Mosaic+Mixup）/Stage2（拡張オフ微調整）。`epoch == stop_epoch` で `best_stg1.pth` をロードし `ema.decay = ema_restart_decay` に設定（`det_solver.py:97-100`）。
- **チェックポイント** ✅: `last.pth` と定期 `checkpoint{epoch:04}.pth`（`(epoch+1)%checkpoint_freq==0`）は **`epoch < stop_epoch`（=Stage1）のときのみ**保存（`det_solver.py:125-131`）。Stage2 は best のみ更新。
- **best選択 + EMA refresh** ✅（`det_solver.py:146-183`、要約より複雑）:
  - 各指標 `k` で `best_stat[k]` を最大追跡。`best_stat[k] > top1` で stage に応じ `best_stg1/best_stg2.pth` 保存。
  - `best_stat['epoch']==epoch`（今回ベスト更新）なら stage に応じ再保存。
  - **`elif epoch >= stop_epoch`（Stage2でベスト未更新の各epoch）→ `best_stat` リセット + `ema.decay -= 0.0001` + `best_stg1.pth` から復元（"Refresh EMA"）**。＝拡張オフ後に改善が出ない間、EMA減衰を段階的に下げつつStage1ベストへ巻き戻す。
- **評価** ✅: 毎epoch `evaluate()` を **EMAモジュール**（`self.ema.module`）で実行（`det_solver.py:133-142`）。COCO eval（faster-coco-eval）。
- **eval成果物** ✅: `eval/latest.pth` を毎epoch、`eval/{epoch:03}.pth` を50epoch毎に保存（`det_solver.py:210-218`）。

**NaN/異常時** ✅（`det_engine.py`、レポート初版の誤りを訂正）:
- `outputs['pred_boxes']` に NaN/Inf → `./NaN.pth` に state を保存するが **訓練は継続**（`:52-62`）。
- **reduced loss が非有限** → `print` して `sys.exit(1)`（`:104-108`）。← 「停止」するのはこちら。

**工夫** — resume/tuning排他、2-stage境界をEMAライフサイクル境界として再利用、複数COCO指標を `top1` に統合して単一bestを選定、異常重みのスナップショット保存。

**弱点/△要確認** — tuning時の score_head 再マッピングの正確な対象/行は未照合（△）。distributed同期は `de_parallel` のラップ剥がし中心。

---

## 3. 最適化・EMA・スケジューラ・DEIM技法（engine/optim, engine/deim）✅optim/ema確認済

### 3.1 LRスケジューラ・warmup・EMA
- **FlatCosineLRScheduler**（`engine/optim/lr_scheduler.py`）: ①warmup（二次 `init_lr*(it/warmup_iter)^2`）→②flat（定数）→③cosine減衰→④no-aug（`min_lr=init_lr*lr_gamma` 固定）。**iter単位**step ⇒ バッチ数/データ量が変わってもLR軌跡が再現可能。
- **LinearWarmup**（`warmup.py`）: `min(1,(step+1)/duration)`、`finished()` 後にメインschedulerへ。
- **ModelEMA** ✅（`engine/optim/ema.py:33-78`）:
  - `deepcopy(de_parallel(model)).eval()` を保持、`requires_grad_(False)`。
  - decay_fn = `warmups==0 ? const : decay*(1-exp(-updates/warmups))`（**指数ランプ**で早期は実効decayが小さく、初期の不安定重みを過剰に焼き付けない）。`start` で最初の `start` 回の update をスキップ。
  - **全 floating-point state_dict（パラメータ＋buffer）** を平均（`:63-66`）。
  - `state_dict={module,updates}`、`load_state_dict(strict=True)` 既定 ✅（`:75`）。
  - 既定 `use_ema=False`, decay0.9999, warmups1000（`DEIM_sandbox/configs/runtime.yml:16-20`）。検証は常にEMA重みで実施（§2）。

### 3.2 最適化器（標準 + 独自2種）✅`optim.py` 確認済
レジストリ（`engine/optim/optim.py`）: `AdamW/SGD/Adam`（torch標準を `register()` でラップ, `:48-50`）に加え:

- **`AutoMuonWithAuxAdam`**（`:53-195`）— Muon と Adam の自動ハイブリッド:
  - パラメータを `ndim ∈ {2,4}`（線形/conv の重み行列）→ **Muon群**、`ndim ∉ {2,4}`（bias/正規化等の1D）→ **Adam群** に自動振り分け（`:71-101`）。
  - Muon既定: `lr=0.01, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5`（Newton-Schulz反復数）。Adam既定: `adam_lr=0.00025, betas=(0.9,0.95), eps=1e-10`。
  - `step()` が群ごとに `muon.muon_update` / `muon.adam_update`（外部 `muon` パッケージ）へ dispatch、decoupled weight decay（`param.mul_(1-lr*wd)`, `:158,193`）。
  - **依存欠落時は `ImportError` を明示送出**（`:132-135,162-165`）＝暗黙fallbackなし。
  - 狙い: Muon は2D/4D重み行列の直交化モメンタム更新で強いが、1D（bias/norm）は恩恵が薄いので Adam に回す。**ndim による自動ルーティングが工夫**。
- **`AdamWScheduleFreeOptimizer`**（`:198-245`）— ScheduleFree をレジストリへ橋渡しする **adapter**:
  - `__new__` が外部 `schedulefree.AdamWScheduleFree(...)` の**インスタンスを返し**、`__init__` は `pass`（no-op）。＝DEIMの `create()` から `type:` 経由で生成できるが、実体は third-party optimizer。**サブクラス化せず外部最適化器をYAML登録系に載せる工夫**。
  - 既定: `lr=0.0025, betas=(0.9,0.999), eps=1e-8, weight_decay=0, warmup_steps=0, r=0.0, weight_lr_power=2.0, foreach=True`。依存欠落時 `ImportError`（`:215-218`）。
  - **ScheduleFree の肝＝train/eval モード切替**: ScheduleFree は LR schedule 不要な代わり、学習step前に `optimizer.train()`、評価/保存前に `optimizer.eval()` が必須。DEIMはこれを **`DetSolver._set_optimizer_mode(mode)`**（`det_solver.py:24-33`）で吸収 — `getattr(optimizer, mode, None)` が callable のときだけ呼ぶ。
    - `_save_state`: eval→保存→train（`:30-33`）＝**checkpointには平均化された eval重みが入る**。
    - `fit()` 全体で train開始 / `evaluate()` 周辺で eval / 直後 train を徹底。
    - 標準optimizer（AdamW/SGD/Muon）には `train/eval` メソッドが無く `mode_fn=None` ⇒ **no-op で安全**。汎用化が工夫。
  - 依存pin: `schedulefree==1.4.1`、`muon-optimizer` は git rev `f98f1ca…` 固定（`pyproject.toml:33-42`）。

### 3.3 DEIM固有損失（engine/deim）
- **Hungarian Matcher**（`matcher.py`）: focal ベースの分類コスト。
- **MAL (Matchability-Aware Loss)**（`deim_criterion.py`）: `target_score=ious^gamma`、`mal_alpha` で前景/背景バランス。
- **GO(union set) マッチング**: 全デコーダ層のマッチ和集合を box/local 損失に適用し収束加速。FGL/DDF損失、`batch_scale=8/batch` 正規化。
- 既定重み/gamma は `configs/base/deim.yml`（`loss_mal:1, loss_bbox:5, loss_giou:2, loss_fgl:0.15, loss_ddf:1.5, gamma:1.5`）。

**△要確認** — Muon/ScheduleFree の実学習での効果切り分け（workdoc に Muon FT best mAP≈0.36 の記録はあるが寄与分離は未検証）。層別LR(正規表現param group)の本sandbox実使用有無。

---

## 4. データ・拡張スケジュールと再現性（engine/data）

- **エポック伝播**: `DetDataset.set_epoch(epoch)` でepoch保持、`transforms(img,target,dataset)` に dataset を渡し拡張がepoch参照可能。`DataLoader.set_epoch` が dataset と collate_fn 両方へ伝播。
- **Compose 3ポリシー**（`data/transforms/container.py`）: `default`/`stop_epoch`(3段 `[start,mid,end]`)/`stop_sample`。`stop_epoch` は `cur_epoch<start` で拡張skip、`>=end` で全拡張off、中間は `mosaic_prob` 制御、Mosaic と ZoomOut/IoUCrop を相互排他。
- **Mosaic**（4枚2×2合成→Affine, `use_cache` で高速化）、**マルチスケール**（`generate_scales` + `BatchImageCollateFunction` が `epoch<stop_epoch` の間バッチ全体を `F.interpolate` で同一スケール化）、**Mixup**（`mixup_prob/mixup_epochs`）。
- **再現性**: `setup_seed(seed)` を **rank別**（`seed+get_rank()`）で random/numpy/torch/cuda に設定、`DistributedSampler.set_epoch` でepoch毎に異なるが再現可能なシャッフル。

**工夫** — 拡張スケジュールの境界 `stop_epoch` を §2/§3 の **EMA再スタート点と共有**し、拡張オフ後の「推論に近い条件」へ学習レジームを移行。

**△要確認** — `setup_seed` の rank加算は report 記載（`engine/misc/dist_utils.py`）で本セッション未照合。`stop_sample` は未使用、mask拡張は一部 `NotImplementedError`。

---

## 5. sandbox独自カスタマイズ・再現環境（patches, uv, docs）✅pyproject確認済

- **早期終了パッチ**（`deim_early_stop.patch` → `det_solver.py:76-86,185-196,220-223`）✅: `early_stop/early_stop_patience/early_stop_min_delta/early_stop_start_epoch` を `yaml_cfg`（CLI `-u` 含む）から取得。指標は val mAP(`top1`)。`top1 > es_best+es_min_delta` で改善、`es_patience` 回連続未改善で `break`。`es_start_epoch` で初期ディップを無視。**disable時は既存挙動不変**。
- **uv再現環境**（`pyproject.toml`）✅: `requires-python ">=3.10,<3.11"`、`torch==2.8.0/torchvision==0.23.0` を `pytorch-cu128` index(`explicit=true`)で固定、`schedulefree==1.4.1`、`muon-optimizer` を git rev pin、`tensorboard>=2.10`、`faster-coco-eval`、`package=false`。`scripts/uv-sync-desktop.sh` が `UV_PROJECT_ENVIRONMENT` を Desktop配下venvに確定。実行は `uv run --no-sync` 推奨（lock再解決回避）。
- **upstream改造patch**: `data_patch.patch`(num_workers 8→4, L4向け)、`new_timm_model.patch`(timm `feature_info.module_name()` の `stages.0`→`stages_0` 明示マッピングで `IntermediateLayerGetter` の RuntimeError 解消)。
- **ドキュメント**: `docs/ONBOARDING.md`(環境再構築/1epoch smoke検証/トラブルシュート)、`docs/LLM_ONBOARDING_SUMMARY.md`(DEIM本体改変禁止・生成物git禁止)。

**工夫** — 早期終了が完全config駆動で非破壊。`uv.lock frozen + index explicit + git pin` で数ヶ月後の別環境でも同一wheelで再構築可能。

**△要確認** — 早期終了patchの submodule への push/PR は未実施（workdoc唯一の未完項目）。

---

## 6. MOTIPとの対比 と 移植候補

| 項目 | DEIM_sandbox | MOTIP(現状) |
|---|---|---|
| 設定管理 | 階層YAML+registry+CLI `-u` dot上書き | 自前YAML(継承/registry限定) |
| TensorBoard | あり(step粒度) | **なし** |
| ロギング | TB + log.txt + config保存 | log.txt のみ(wandb任意) |
| チェックポイント | last/best_stg1/best_stg2/定期 | 単純(two-stage best なし) |
| EMA | あり(指数ランプ, 検証はEMA) | **なし** |
| AMP | あり(forward FP16/loss FP32) | **なし** |
| スケジューラ | FlatCosine(iter単位)+warmup | 簡素 |
| 早期終了 | あり(val mAP, config駆動) | **なし** |
| 最適化器 | AdamW/SGD/Muon/ScheduleFree差替 | 限定 |
| 再現環境 | uv.lock frozen+cu128固定+seed(rank別) | lock/index固定は前提でない |

**MOTIPへ移植すると有用（優先度順）**
1. **TensorBoard + log.txt 二層ロギング**（`det_engine.py:112-117`）: 既存log.txtを壊さずstep粒度の loss/lr 可視化を追加。
2. **config駆動の早期終了**（`deim_early_stop.patch` の設計）: 監視指標を MOTIP の主要metric（HOTA/MOTA等）に差し替えれば非破壊導入可。`start_epoch/min_delta/patience` のwarmup・plateau対策がそのまま有用。
3. **AMP（forward FP16 / loss FP32 分離）**: メモリ/速度改善、loss FP32で数値安定。
4. **EMA（指数ランプdecay, 検証はEMA重み）**: 検証安定化（ID追跡系への有効性は要評価）。
5. **uv.lock frozen + index explicit + git pin**: 版揺れ排除。
6. **CLI `-u` dot上書き**: 新configファイルを作らず実験バリエーション増加、実行コマンドが再現記録に。

---

## 7. △要確認・追加調査リスト
1. configツリーの参照系統: 直下 `configs/` と `DEIM/configs/` のどちらが実行時に使われるか（相対パス依存）と差分。
2. tuning時 score_head 再マッピングの対象パラメータ/行（`_solver.py`）。
3. `setup_seed` の rank加算（`engine/misc/dist_utils.py`）— report記載のみ、本セッション未照合。
4. 層別LR（正規表現param group）の本sandbox実使用有無。
5. Muon/ScheduleFree の実効果切り分け、fine-tune最終収束特性。
6. `best選択ブロック`（`det_solver.py:158-183`）の入れ子条件の網羅的意味論検証。

---

---

## 8. 参考コード 絶対パス索引（対応箇所）

| トピック | 絶対パス:行 | 本書の対応節 |
|---|---|---|
| エントリ（config構築/solver選択） | `/workspace/Project/DEIM_sandbox/DEIM/train.py:28-53` | §1,§2 |
| YAML合成/CLIパース | `/workspace/Project/DEIM_sandbox/DEIM/engine/core/yaml_utils.py` | §1 |
| registry/create | `/workspace/Project/DEIM_sandbox/DEIM/engine/core/workspace.py` | §1 |
| Solver fit / best選択 / EMA refresh / 保存 | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:35-227`（保存 `:125-183`, mode切替 `:24-33`） | §2,§3 |
| 早期終了パッチ適用後 | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py:76-86,185-196,220-223` | §5 |
| Engine train/eval / AMP / NaN / TB / iter-step | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py:24-177`（AMP `:48-76`, NaN `:52-62,104-108`, TB `:112-117`, step `:95-99`） | §2,§3 |
| Solver基底 setup/resume/tuning/state_dict | `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/_solver.py` | §2 |
| EMA | `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78` | §3.1 |
| 最適化器(Muon/ScheduleFree/標準) | `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/optim.py:48-251`（Muon `:53-195`, ScheduleFree `:198-245`） | §3.2 |
| LRスケジューラ/warmup | `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/{lr_scheduler.py,warmup.py}` | §3.1 |
| DEIM損失/マッチング(MAL等) | `/workspace/Project/DEIM_sandbox/DEIM/engine/deim/{deim_criterion.py,matcher.py}` | §3.3 |
| データ/拡張スケジュール | `/workspace/Project/DEIM_sandbox/DEIM/engine/data/{dataloader.py,transforms/container.py,transforms/mosaic.py,dataset/_dataset.py}` | §4 |
| seed/分散 | `/workspace/Project/DEIM_sandbox/DEIM/engine/misc/dist_utils.py`（△未照合） | §4 |
| 共通既定(runtime) | `/workspace/Project/DEIM_sandbox/configs/runtime.yml` | §0.2,§2,§3 |
| base(model/最適化/拡張/損失) | `/workspace/Project/DEIM_sandbox/configs/base/deim.yml` | §3.3 |
| 再現環境 | `/workspace/Project/DEIM_sandbox/pyproject.toml`, `/workspace/Project/DEIM_sandbox/scripts/uv-sync-desktop.sh` | §5 |
| 独自patch | `/workspace/Project/DEIM_sandbox/{deim_early_stop.patch,data_patch.patch,new_timm_model.patch}` | §5 |
| 比較対象 MOTIP | `/home/kasm-user/Desktop/MOTIP/{train.py,log/logger.py,configs/,pyproject.toml}` | §6 |

### 確認済み主張の根拠（本セッションでコーディネータが直接読んで照合した絶対パス）
- `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_solver.py`（全242行）
- `/workspace/Project/DEIM_sandbox/DEIM/engine/solver/det_engine.py`（全177行）
- `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/optim.py`（全251行）
- `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py`（全103行）
- `/workspace/Project/DEIM_sandbox/DEIM/train.py`（冒頭〜引数定義）
- `/workspace/Project/DEIM_sandbox/configs/runtime.yml`（全20行）
- `/workspace/Project/DEIM_sandbox/pyproject.toml`（全52行）

上記以外（`engine/core/*`, `engine/data/*`, `engine/deim/*`, `engine/misc/dist_utils.py`, 各 `*.patch`, `docs/*`）はサブエージェント調査に基づく記述で、`△要確認` を付した箇所はコーディネータ未照合。
