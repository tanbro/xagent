# 上传文件路径迁移工具

## 背景

`uploaded_files` 表中的 `storage_path` 字段需要在不同格式之间转换：

- **绝对路径**：`/uploads/user_1/web_task_123/output/file.txt`
- **相对路径**：`web_task_123/output/file.txt`（不含 `user_{user_id}` 前缀）

## 核心功能：处理多个 uploads_dir

**此工具主要用于处理 `XAGENT_UPLOADS_DIR` 配置被多次修改的情况。**

当 uploads 目录位置改变后，数据库中会存在多个不同前缀的路径：

```
/old/location/uploads/user_1/task_1/file.txt    ← 旧的 uploads_dir
/new/location/uploads/user_1/task_2/file.txt    ← 中间的 uploads_dir
/current/uploads/user_1/task_3/file.txt         ← 当前的 uploads_dir
```

Alembic 自动迁移只能处理**当前配置的 `UPLOADS_DIR`**，其他路径需要手动指定源目录进行转换。

**使用 `-d` 参数指定旧的 uploads 目录路径**：
```bash
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/location/uploads --confirm
```

**可以指定多个源目录**（按时间顺序从旧到新）：
```bash
python scripts/migrate_uploads_file_abs_path.py migrate \
  -d /oldest/uploads \
  -d /middle/uploads \
  -d /old/uploads \
  --confirm
```

**示例**：假设你的 uploads 目录经历了三次迁移：

```
/var/www/xagent/uploads          ← 2023年使用
/home/user/xagent/uploads        ← 2024年迁移到这
/mnt/d/work/xagent/uploads        ← 2025年迁移到这
/current/xagent/uploads          ← 当前配置
```

数据库中可能存在这样的路径：

```
/var/www/xagent/uploads/user_1/task_1/file.txt
/home/user/xagent/uploads/user_1/task_2/file.txt
/mnt/d/work/xagent/uploads/user_1/task_3/file.txt
```

运行命令将它们统一转换为相对路径：

```bash
python scripts/migrate_uploads_file_abs_path.py migrate \
  -d /var/www/xagent/uploads \
  -d /home/user/xagent/uploads \
  -d /mnt/d/work/xagent/uploads \
  --confirm
```

转换后都变成：

```
task_1/file.txt
task_2/file.txt
task_3/file.txt
```

## 何时使用此工具

### 情况 1：Alembic 迁移失败

当运行 `alembic upgrade` 或 `alembic downgrade` 时，如果看到类似警告：

```
Warning: Could not convert /some/path to relative path (outside UPLOADS_DIR): ...
Some paths could not be converted automatically.
Please use the manual migration tool to fix these paths
```

说明部分文件路径不在当前的 `UPLOADS_DIR` 下，自动转换失败。此时需要使用此工具手动处理。

### 情况 2：uploads 目录多次迁移

如果 `XAGENT_UPLOADS_DIR` 配置被多次修改，数据库中可能存在多个不同前缀的路径：

```
/old/uploads/user_1/task_1/file.txt
/new/uploads/user_1/task_2/file.txt
/current/uploads/user_1/task_3/file.txt
```

自动迁移只能处理当前配置的目录，其他路径需要手动指定源目录进行转换。

### 情况 3：混合路径格式

数据库中同时存在绝对路径和相对路径，需要统一格式。

## 使用方法

### 1. 检查当前状态

```bash
python scripts/migrate_uploads_file_abs_path.py check
```

输出示例：

```
Migration Status
┌────────────────────────────────────────────┬───────┬────────────┐
│ Category                                   │ Count │ Percentage │
├────────────────────────────────────────────┼───────┼────────────┤
│ Total records                              │    36 │            │
│ Absolute paths (need migration)            │    30 │    83.3%   │
│ Relative paths (already migrated)          │     6 │    16.7%   │
└────────────────────────────────────────────┴───────┴────────────┘

⚠ 30 absolute paths need migration.
```

### 2. 预览迁移（dry-run 模式）

**默认是 dry-run 模式，不会修改数据库**，只会显示将要进行的转换。

```bash
# 指定旧的 uploads 目录（不带 --confirm 就是预览）
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads

# 支持指定多个源目录
python scripts/migrate_uploads_file_abs_path.py migrate -d /old1/uploads -d /old2/uploads
```

输出示例：

```
Migration Summary
┌──────────────────────────────┬───────┬───────┐
│ Category                     │ Count │ Status│
├──────────────────────────────┼───────┼───────┤
│ Total scanned                │    30 │       │
│ To migrate                   │    30 │   ✓   │
└──────────────────────────────┴───────┴───────┘

Preview (showing 10/30 records that will be migrated):
📄 c42346a0...
   Before: /old/uploads/user_1/web_task_25/output/chart.html
   After:  web_task_25/output/chart.html

ℹ️  DRY RUN MODE - No changes made to database
Use --confirm to actually migrate these records.
```

### 3. 执行迁移

**添加 `--confirm` 参数才会真正修改数据库**。

```bash
# 确认后执行迁移
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads --confirm
```

输出示例：

```
Migration Summary
┌──────────────────────────────┬───────┬───────┐
│ Category                     │ Count │ Status│
├──────────────────────────────┼───────┼───────┤
│ To migrate                   │    30 │   ✓   │
└──────────────────────────────┴───────┴───────┘

✓ Successfully migrated 30 records
```

### 4. 高级选项

```bash
# 显示所有记录（默认只显示前 10 条）
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads -v

# 自定义批处理大小（默认 1000）
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads -b 500
```

## 常见问题

### Q: 如何找到旧的 uploads 目录？

A: 查看数据库中的路径：

```bash
sqlite3 xagent.db "SELECT DISTINCT substr(storage_path, 1, 50) FROM uploaded_files LIMIT 10;"
```

或者使用 check 命令查看路径样本。

### Q: 迁移后还需要运行 Alembic 吗？

A: 取决于你的目标状态：

- **目标是相对路径**：手动迁移后，运行 `alembic upgrade head` 更新版本号
- **目标是绝对路径**：手动迁移后，运行 `alembic downgrade -1` 回退版本号

### Q: 迁移会失败吗？

A: 如果路径不在指定的 `--uploads-dir` 下，会被跳过并统计在 "Other absolute paths" 中。添加更多 `-d` 选项来覆盖所有路径。

### Q: 如何回滚？

A: 此脚本不会备份，建议先备份数据库：

```bash
cp xagent.db xagent.db.backup
```

## 技术说明

### 路径格式规则

- **相对路径**：不含 `user_{user_id}` 前缀，如 `web_task_123/output/file.txt`
- **绝对路径**：完整路径，如 `/uploads/user_1/web_task_123/output/file.txt`

### 跨平台支持

脚本会自动检测 Windows 风格路径（`C:\...`）和 Unix 风格路径，使用对应的 Path 类进行处理。

### 性能优化

- 使用 `yield_per` 流式处理，避免内存溢出
- 批量提交（默认每 1000 条），提高性能
- 支持大型数据库（数百万条记录）
