# P0/P1 性能等效迁移记录 — 2026-08-03

## 结果
- 新代码（reference P0/P1，warm-up 指纹 8410ae8b）已上线，3 个 worker 全部 RUNNING。
- 监督完成 ≥2 轮：WL4 162→165、WL5 164→167、WL6 164→168（全部为迁移后新代码完成的窗口，指纹 8410ae8b）。
- 无 API 错误（429/503=0）、无 Traceback、无迁移拒绝。

## 进度保存
- 迁移前完整备份：`_progress_backup_20260803_200911/`（results_fm 900MB + runtime_state wl4-6 + scripts_old）。
- 迁移脚本自动备份：`runtime/state/performance_equivalence_backups/20260803T121143Z/`（WL4-6 全部历史 window_state 原始 b93 版本）。

## 迁移步骤（无损）
1. 停旧 worker（stop.ps1）。
2. 备份当前 WL4-6 结果 + runtime/state + scripts。
3. 覆盖 3 个源码文件：scheduler/run_pipeline.py、factorminer/core/ralph_loop.py、factorminer/data/preprocessor.py（内容哈希与证书 target 一致：scheduler=d818a349）。
4. 复制新 warmup 8410ae8b + performance_equivalence 证书到 bundle。
5. 更新 scripts：migrate_window_states.py、verify_bundle.py（期望 8410）、fm_worker.ps1（加迁移预检 + --fm-performance-equivalent-from，保留 PYTHONUTF8/UTF-8 修复）。
6. 保留 portable_runner.py 的 POSIX 路径归一化 shim（Windows 反斜杠会让指纹算错；reference 干净版在 Windows 会 fail）。
7. 运行 migrate_window_states.py：WL4 162/162、WL5 164/164、WL6 164/164 窗口指纹 b93→8410（幂等，已备份原始文件）。
8. 离线验证：OFFLINE VERIFY OK WL[4..9] share completed warm-up 8410ae8b。
9. start.ps1 重启，新 fm_worker 自动预检迁移（no-op）并以 --fm-performance-equivalent-from 启动。

## 证据
- 新窗口 window_state: warmup_fingerprint=8410ae8b，performance_equivalence 标记仅存在于迁移改写的旧窗口。
- forward: WL4 nav=915,464 / 165 decisions；WL5 nav=1,711,443 / 167；WL6 nav=1,324,237 / 168。
- scheduler/run_pipeline.py sha256 = d818a349...（证书 target_scheduler_code_sha256 一致）。
