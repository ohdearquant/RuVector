// Self-contained runtime smoke test for #657 (query arg forms) and #658
// (create-on-existing error). No native addon required: #657 uses a mock
// backend via RvfDatabase.fromBackend(); #658 stubs @ruvector/rvf-node with a
// no-op native so NodeBackend.create() reaches its precondition check.
//
// Run: node scripts/smoke-query-create.mjs   (after `tsc` has emitted dist/)
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const dist = pathToFileURL(path.resolve('dist/index.js')).href;
const { RvfDatabase, RvfError, RvfErrorCode } = await import(dist);

let passed = 0;
const ok = (name) => { console.log(`  ok - ${name}`); passed++; };

// ---- #657: query() accepts count as number OR {k|topK|limit}, validates ----
function mockBackend() {
  const calls = [];
  return {
    _calls: calls,
    query: async (_v, count, options) => { calls.push({ count, options }); return []; },
  };
}

{
  const be = mockBackend();
  const db = RvfDatabase.fromBackend(be);
  const v = new Float32Array([1, 0, 0, 0]);

  await db.query(v, 5);
  assert.strictEqual(be._calls.at(-1).count, 5, 'positional number');
  ok('#657 positional count query(v, 5) -> 5');

  await db.query(v, { k: 2 });
  assert.strictEqual(be._calls.at(-1).count, 2, '{k}');
  ok('#657 object {k:2} -> 2');

  await db.query(v, { topK: 3 });
  assert.strictEqual(be._calls.at(-1).count, 3, '{topK}');
  ok('#657 object {topK:3} -> 3');

  await db.query(v, { limit: 4 });
  assert.strictEqual(be._calls.at(-1).count, 4, '{limit}');
  ok('#657 object {limit:4} -> 4');

  await db.query(v, { k: 7, efSearch: 200 });
  assert.strictEqual(be._calls.at(-1).count, 7);
  assert.strictEqual(be._calls.at(-1).options.efSearch, 200, 'object options passthrough');
  ok('#657 object {k, efSearch} forwards efSearch');

  await db.query(v, { k: 6, topK: 6, limit: 6 });
  assert.strictEqual(be._calls.at(-1).count, 6, 'matching aliases');
  ok('#657 matching count aliases are accepted');

  for (const [arg, label] of [
    [{}, 'no count'],
    [{ k: 2, topK: 3 }, 'conflicting aliases'],
    [-1, 'negative'],
    [2.5, 'non-integer'],
    [0, 'zero'],
    [0x1_0000_0000, 'larger than u32'],
  ]) {
    await assert.rejects(
      () => db.query(v, arg),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.InvalidArgument,
      `should reject ${label} with InvalidArgument`,
    );
    ok(`#657 rejects ${label} with clear InvalidArgument (not napi error)`);
  }
}

// ---- #658: create() on an existing path -> clear FileExists, or overwrite ----
// Requires a stub @ruvector/rvf-node resolvable from dist/backend.js (i.e.
// installed in this package's node_modules). The runner script sets that up;
// if it is absent we skip rather than fail.
{
  let nativeAvailable = true;
  try { await import('@ruvector/rvf-node'); } catch { nativeAvailable = false; }

  if (!nativeAvailable) {
    console.log('  skip - #658 (no @ruvector/rvf-node stub resolvable; run via npm run smoke)');
  } else {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-smoke-'));
    const target = path.join(tmp, 'library.rvf');
    fs.writeFileSync(target, 'pretend existing rvf content');

    // Existing file, no overwrite -> FileExists (NOT FsyncFailed)
    await assert.rejects(
      () => RvfDatabase.create(target, { dimensions: 4 }, 'node'),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.FileExists && !/fsync/i.test(e.message),
      'create() on existing file should throw FileExists, not FsyncFailed',
    );
    ok('#658 create() on existing path -> FileExists (clear, not FsyncFailed)');

    // overwrite:true removes the old file (+ its sidecar) then proceeds
    fs.writeFileSync(`${target}.idmap.json`, '{"stale":true}');
    const replaced = await RvfDatabase.create(target, { dimensions: 4, overwrite: true }, 'node');
    assert.ok(!fs.existsSync(`${target}.idmap.json`), 'overwrite should clear stale sidecar');
    ok('#658 create({overwrite:true}) clears old file + sidecar and proceeds');
    await replaced.close();

    // A failed replacement restores both original files instead of destroying
    // the store before the native create succeeds.
    const rollbackTarget = path.join(tmp, 'rollback.rvf');
    const rollbackSidecar = `${rollbackTarget}.idmap.json`;
    fs.writeFileSync(rollbackTarget, 'original-store');
    fs.writeFileSync(rollbackSidecar, 'original-sidecar');
    await assert.rejects(
      () => RvfDatabase.create(rollbackTarget, { dimensions: 0, overwrite: true }, 'node'),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.InvalidOptions,
      'invalid replacement should fail before losing the original store',
    );
    assert.strictEqual(fs.readFileSync(rollbackTarget, 'utf-8'), 'original-store');
    assert.strictEqual(fs.readFileSync(rollbackSidecar, 'utf-8'), 'original-sidecar');
    ok('#658 failed overwrite restores the original store and sidecar');

    const orphanTarget = path.join(tmp, 'orphan.rvf');
    fs.writeFileSync(`${orphanTarget}.idmap.json`, '{"orphan":true}');
    await assert.rejects(
      () => RvfDatabase.create(orphanTarget, { dimensions: 4 }, 'node'),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.FileExists,
      'an orphaned ID-map sidecar must block a fresh create',
    );
    ok('#658 orphaned sidecar is not silently reused');

    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

console.log(`\nAll ${passed} checks passed.`);
