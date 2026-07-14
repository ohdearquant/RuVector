// Runtime smoke test for #635: the .idmap.json sidecar must be written
// atomically and must fail loud (not degrade to empty maps) on a corrupt load.
// Requires a stub @ruvector/rvf-node resolvable from dist/backend.js (the
// runner sets one up); skips if absent.
//
// Run: node scripts/smoke-idmap-durability.mjs   (after `tsc` emits dist/)
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const dist = pathToFileURL(path.resolve('dist/index.js')).href;
const { RvfDatabase, RvfError, RvfErrorCode } = await import(dist);

let passed = 0;
const ok = (name) => { console.log(`  ok - ${name}`); passed++; };

let nativeAvailable = true;
try { await import('@ruvector/rvf-node'); } catch { nativeAvailable = false; }
if (!nativeAvailable) {
  console.log('  skip - #635 (no @ruvector/rvf-node stub resolvable; run via npm run smoke:idmap)');
  process.exit(0);
}

const vec = () => new Float32Array([1, 0, 0, 0]);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-idmap-'));
const target = path.join(tmp, 'store.rvf');
const sidecar = `${target}.idmap.json`;

// 1. Ingest with string ids -> sidecar written atomically (no leftover .tmp).
{
  const db = await RvfDatabase.create(target, { dimensions: 4 }, 'node');
  await db.ingestBatch([
    { id: 'a', vector: vec() },
    { id: 'b', vector: vec() },
  ]);
  await db.close();

  const sc = JSON.parse(fs.readFileSync(sidecar, 'utf-8'));
  assert.strictEqual(sc.nextLabel, 3, 'two ids -> labels 1,2 -> nextLabel 3');
  assert.ok(!fs.existsSync(`${sidecar}.tmp`), 'atomic write leaves no .tmp file');
  ok('#635 ingest persists sidecar atomically (nextLabel=3, no .tmp)');
}

// 2. Reopen with a VALID sidecar restores state (nextLabel does NOT reset).
{
  const db = await RvfDatabase.open(target, 'node');
  await db.ingestBatch([{ id: 'c', vector: vec() }]);
  await db.close();

  const sc = JSON.parse(fs.readFileSync(sidecar, 'utf-8'));
  assert.strictEqual(sc.nextLabel, 4, 'restored state -> new id gets label 3 -> nextLabel 4 (not reset to 2)');
  assert.deepStrictEqual(Object.keys(sc.idToLabel).sort(), ['a', 'b', 'c']);
  ok('#635 reopen restores maps; no label collision on next ingest');
}

// 3. A corrupt sidecar fails loud (SidecarCorrupt) and is quarantined, not
//    overwritten with empty/colliding state.
{
  fs.writeFileSync(sidecar, '{ this is not valid json ', 'utf-8');
  await assert.rejects(
    () => RvfDatabase.open(target, 'node'),
    (e) =>
      e instanceof RvfError &&
      e.code === RvfErrorCode.SidecarCorrupt &&
      !/fsync/i.test(e.message),
    'open() on corrupt sidecar should throw SidecarCorrupt',
  );
  assert.ok(!fs.existsSync(sidecar), 'corrupt sidecar renamed away (not left to be overwritten)');
  const quarantined = fs.readdirSync(tmp).filter((f) => f.includes('.idmap.json.corrupt-'));
  assert.strictEqual(quarantined.length, 1, 'exactly one quarantine file created');
  ok('#635 corrupt sidecar -> SidecarCorrupt + quarantined (no silent empty-map reset)');
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\nAll ${passed} checks passed.`);
