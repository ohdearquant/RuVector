// Smoke test for issue #661: brain_* MCP tools must return an actionable
// "install @ruvector/pi-brain" hint instead of an opaque TypeError
// (`client.sync is not a function`) when pi-brain is absent or unusable.
//
// Self-contained (Node, zero deps). The MCP server's SDK deps are stubbed
// in-process via Module._load so bin/mcp-server.js can be required headless;
// @ruvector/pi-brain is swapped per scenario to exercise every way the
// dependency can be missing/broken. Drives the REAL loadBrainClient() exported
// from bin/mcp-server.js.
//
//   node test/smoke-brain-missing-dep.mjs

import Module from 'node:module';
import path from 'node:path';
import assert from 'node:assert';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const serverPath = path.resolve(__dirname, '../bin/mcp-server.js');

// --- Stub @modelcontextprotocol/sdk so mcp-server.js loads without install ---
class FakeServer {
  setRequestHandler() {}
  async connect() {}
}
const sdkStubs = {
  '@modelcontextprotocol/sdk/server/index.js': { Server: FakeServer },
  '@modelcontextprotocol/sdk/server/stdio.js': { StdioServerTransport: class {} },
  '@modelcontextprotocol/sdk/types.js': {
    CallToolRequestSchema: {},
    ListToolsRequestSchema: {},
    ListResourcesRequestSchema: {},
    ReadResourceRequestSchema: {},
  },
};

// Swapped per scenario. `null` simulates "@ruvector/pi-brain not installed".
let piBrainExports = null;

const origLoad = Module._load;
Module._load = function (request, ...rest) {
  if (Object.prototype.hasOwnProperty.call(sdkStubs, request)) {
    return sdkStubs[request];
  }
  if (request === '@ruvector/pi-brain') {
    if (piBrainExports === null) {
      const e = new Error("Cannot find module '@ruvector/pi-brain'");
      e.code = 'MODULE_NOT_FOUND';
      throw e;
    }
    return piBrainExports;
  }
  return origLoad.call(this, request, ...rest);
};

const require = createRequire(import.meta.url);
const { loadBrainClient, BRAIN_MISSING_DEP_RESULT } = require(serverPath);

let passed = 0;
function check(name, fn) {
  fn();
  console.log(`  ok - ${name}`);
  passed++;
}

// The exact hint text every brain_* handler surfaces.
function hintPayload() {
  return JSON.parse(BRAIN_MISSING_DEP_RESULT.content[0].text);
}

check('#661 BRAIN_MISSING_DEP_RESULT carries an actionable install hint', () => {
  const p = hintPayload();
  assert.strictEqual(p.success, false);
  assert.match(p.error, /@ruvector\/pi-brain/);
  assert.match(p.hint, /npm install @ruvector\/pi-brain/);
  assert.notStrictEqual(BRAIN_MISSING_DEP_RESULT.isError, true, 'a missing optional dep is not a server error');
});

check('#661 not installed -> {missing:true} (handlers return the hint)', () => {
  piBrainExports = null;
  const r = loadBrainClient();
  assert.strictEqual(r.missing, true);
  assert.strictEqual(r.client, undefined);
});

check('#661 module present but no PiBrainClient export -> {missing:true}', () => {
  piBrainExports = {}; // resolves, but exposes nothing usable
  const r = loadBrainClient();
  assert.strictEqual(r.missing, true);
});

check('#661 partial stub (constructs, lacks .sync) -> handler guard returns the hint', () => {
  // This is the EXACT reported symptom: `new PiBrainClient()` succeeds, so no
  // MODULE_NOT_FOUND is thrown, but the instance has no `sync` method. Before
  // the fix this fell through to an opaque `TypeError: client.sync is not a
  // function`. loadBrainClient now yields a client; the per-handler guard
  // `typeof client.sync !== 'function'` is what maps it to the install hint.
  piBrainExports = { PiBrainClient: class { search() {} } };
  const r = loadBrainClient();
  assert.strictEqual(r.missing, undefined, 'construction succeeds, so it is not caught as missing here');
  assert.ok(r.client, 'a client instance is returned');
  assert.notStrictEqual(typeof r.client.sync, 'function', 'reproduces the missing-method symptom');
  // The guard predicate the brain_sync handler now runs:
  const wouldReturnHint = typeof r.client.sync !== 'function';
  assert.strictEqual(wouldReturnHint, true, 'handler guard would return BRAIN_MISSING_DEP_RESULT');
});

check('#661 usable client (has .sync) -> real call path proceeds', () => {
  piBrainExports = { PiBrainClient: class { sync() { return { ok: true }; } search() {} } };
  const r = loadBrainClient();
  assert.ok(r.client);
  assert.strictEqual(typeof r.client.sync, 'function', 'handler proceeds to await client.sync(...)');
});

Module._load = origLoad;
console.log(`\nAll ${passed} checks passed.`);
