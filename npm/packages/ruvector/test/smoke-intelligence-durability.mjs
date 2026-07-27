// Runtime smoke test for #634: the .ruvector/intelligence.json store must be
// written atomically and must fail loud (not silently wipe) on a corrupt read.
// Drives the REAL functions exported from bin/cli.js (commander/chalk are
// stubbed only so the module can be required without the full CLI install).
//
// Run: node test/smoke-intelligence-durability.mjs
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const cliPath = path.join(here, '..', 'bin', 'cli.js');

// --- stub the two module-scope deps so cli.js can be required as a library ---
const stubRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ruv634-stubs-'));
function writeStub(name, indexJs) {
  const dir = path.join(stubRoot, 'node_modules', name);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name, version: '0.0.0', main: 'index.js' }));
  fs.writeFileSync(path.join(dir, 'index.js'), indexJs);
}
// commander: chainable no-op Command — any method returns a chainable, and
// command()/createCommand() return fresh chainables.
writeStub('commander', `
function makeCmd() {
  const cmd = new Proxy(function () {}, {
    get(_t, prop) {
      if (prop === 'command' || prop === 'createCommand' || prop === 'addCommand') return () => makeCmd();
      if (prop === 'opts' || prop === 'optsWithGlobals') return () => ({});
      if (prop === 'args') return [];
      return () => cmd;
    },
    apply() { return makeCmd(); },
  });
  return cmd;
}
class Command { constructor() { return makeCmd(); } }
module.exports = { Command, program: makeCmd() };
`);
// chalk: identity color functions
writeStub('chalk', `
const id = (s) => s;
const handler = { get: () => new Proxy(id, handler), apply: (t, _th, a) => a[0] };
const chalk = new Proxy(id, handler);
module.exports = chalk; module.exports.default = chalk;
`);

// Make the stubs resolvable from cli.js by pointing NODE_PATH — but require()
// resolves relative to cli.js, so instead patch module resolution via a temp
// symlinked node_modules next to the package.
const pkgNodeModules = path.join(here, '..', 'node_modules');
let createdSymlink = false;
if (!fs.existsSync(pkgNodeModules)) {
  fs.symlinkSync(path.join(stubRoot, 'node_modules'), pkgNodeModules, 'dir');
  createdSymlink = true;
}

let passed = 0;
const ok = (name) => { console.log(`  ok - ${name}`); passed++; };

try {
  const { atomicWriteFileSync, readIntelStoreSafe, Intelligence } = require(cliPath);

  // ---- helper: atomicWriteFileSync leaves no temp + produces the exact bytes
  {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruv634-aw-'));
    const f = path.join(dir, 'x.json');
    atomicWriteFileSync(f, '{"a":1}');
    assert.strictEqual(fs.readFileSync(f, 'utf-8'), '{"a":1}');
    assert.deepStrictEqual(fs.readdirSync(dir), ['x.json'], 'no .tmp file left behind');
    ok('#634 atomicWriteFileSync writes exact bytes, no temp leftover');
    fs.rmSync(dir, { recursive: true, force: true });
  }

  // ---- helper: readIntelStoreSafe absent -> {}, valid -> parsed, corrupt -> throw+quarantine
  {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruv634-rd-'));
    const f = path.join(dir, 'intelligence.json');
    assert.deepStrictEqual(readIntelStoreSafe(f), {}, 'absent store -> {}');
    fs.writeFileSync(f, JSON.stringify({ memories: [1, 2, 3] }));
    assert.deepStrictEqual(readIntelStoreSafe(f).memories, [1, 2, 3], 'valid store parsed');
    fs.writeFileSync(f, '{ torn ');
    assert.throws(() => readIntelStoreSafe(f), /corrupt and was quarantined/, 'corrupt store throws');
    assert.ok(!fs.existsSync(f), 'corrupt store renamed away (not left to be overwritten)');
    assert.strictEqual(fs.readdirSync(dir).filter((x) => x.includes('.corrupt-')).length, 1, 'one quarantine file');
    ok('#634 readIntelStoreSafe: absent->{}, valid->parsed, corrupt->throw+quarantine');
    fs.rmSync(dir, { recursive: true, force: true });
  }

  // ---- the reported wipe scenario, through the real Intelligence class ----
  {
    const proj = fs.mkdtempSync(path.join(os.tmpdir(), 'ruv634-proj-'));
    const store = path.join(proj, '.ruvector', 'intelligence.json');
    fs.mkdirSync(path.dirname(store), { recursive: true });
    // A store with 3 memories that must survive.
    fs.writeFileSync(store, JSON.stringify({
      patterns: {}, memories: ['m1', 'm2', 'm3'], trajectories: [], errors: {},
      file_sequences: [], agents: {}, edges: [],
      stats: { total_patterns: 0, total_memories: 3, total_trajectories: 0, total_errors: 0, session_count: 1, last_session: 1 },
    }, null, 2));

    const prevCwd = process.cwd();
    process.chdir(proj);
    try {
      // Sanity: a valid store loads its 3 memories.
      const okIntel = new Intelligence({ skipEngine: true });
      assert.strictEqual(okIntel.data.memories.length, 3, 'valid store loads 3 memories');

      // Corrupt the store exactly as the issue's repro does (truncated file).
      const full = fs.readFileSync(store, 'utf-8');
      fs.writeFileSync(store, full.slice(0, Math.floor(full.length / 2)));

      // OLD behavior: construct succeeds, load() returns empty defaults, next
      // save() persists the emptiness (wipe). NEW behavior: construct throws.
      assert.throws(() => new Intelligence({ skipEngine: true }), /corrupt and was quarantined/,
        'corrupt store -> throw, not silent empty-defaults');
      assert.ok(!fs.existsSync(store), 'corrupt store quarantined (not left in place to be overwritten)');
      const quarantined = fs.readdirSync(path.dirname(store)).filter((x) => x.includes('.corrupt-'));
      assert.strictEqual(quarantined.length, 1, 'store quarantined once (data preserved for recovery)');
      // The quarantined file still holds the original (now-truncated) bytes, not an empty store.
      ok('#634 corrupt store is quarantined + fails loud instead of silently wiping memories');

      // Atomic save roundtrip: fresh store, save leaves no temp, reload matches.
      const fresh = new Intelligence({ skipEngine: true });
      fresh.data.memories.push('kept');
      fresh.save();
      assert.ok(!fs.readdirSync(path.dirname(store)).some((x) => x.includes('.tmp.')), 'save() leaves no temp file');
      const reloaded = new Intelligence({ skipEngine: true });
      assert.ok(reloaded.data.memories.includes('kept'), 'saved memory survives reload');
      ok('#634 save() is atomic (no temp leftover) and roundtrips');
    } finally {
      process.chdir(prevCwd);
    }
    fs.rmSync(proj, { recursive: true, force: true });
  }

  console.log(`\nAll ${passed} checks passed.`);
} finally {
  if (createdSymlink) { try { fs.unlinkSync(pkgNodeModules); } catch { /* ignore */ } }
  fs.rmSync(stubRoot, { recursive: true, force: true });
}
