#!/usr/bin/env node
/**
 * CI smoke test for the ESM entry point (#656).
 *
 * Regresses the bug where `dist/esm/*.js` emitted extensionless relative
 * specifiers, which made `import('@ruvector/ruvllm')` throw
 * `ERR_MODULE_NOT_FOUND` under Node's native ESM resolver.
 *
 * This packs the current working tree with `npm pack`, installs the
 * resulting tarball into a scratch directory whose own package.json
 * declares `"type": "module"`, then dynamically imports the package
 * exactly the way a real ESM consumer would and asserts the default
 * export exists.
 *
 * NOTE: this requires the native napi `.node` addon to be present (built by
 * the `build:native` npm script / the platform CI job) so that loading the
 * compiled engine at import time succeeds. It is meant to run in CI after
 * the native build step, not necessarily in a bare local checkout without
 * a prebuilt addon.
 */

import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const PKG_ROOT = fileURLToPath(new URL('..', import.meta.url));

function main() {
  console.log('smoke-esm: packing package with `npm pack`...');
  const packOutput = execFileSync('npm', ['pack', '--json'], {
    cwd: PKG_ROOT,
    encoding: 'utf8',
  });
  const [{ filename }] = JSON.parse(packOutput);
  const tarballPath = join(PKG_ROOT, filename);

  const scratch = mkdtempSync(join(tmpdir(), 'ruvllm-esm-smoke-'));

  try {
    writeFileSync(
      join(scratch, 'package.json'),
      JSON.stringify(
        { name: 'ruvllm-esm-smoke', version: '0.0.0', private: true, type: 'module' },
        null,
        2
      )
    );

    console.log('smoke-esm: installing tarball into scratch dir (as a "type":"module" consumer)...');
    execFileSync(
      'npm',
      ['install', '--no-audit', '--no-fund', '--no-package-lock', '--no-save', tarballPath],
      { cwd: scratch, stdio: 'inherit' }
    );

    console.log('smoke-esm: dynamically importing @ruvector/ruvllm...');
    const assertionScript = [
      "import('@ruvector/ruvllm').then((mod) => {",
      "  if (typeof mod.default === 'undefined') {",
      "    console.error('smoke-esm: FAIL - default export is undefined');",
      '    process.exit(1);',
      '  }',
      "  console.log('smoke-esm: OK - import() resolved, default export is', typeof mod.default);",
      '}).catch((err) => {',
      "  console.error('smoke-esm: FAIL -', (err && err.stack) || err);",
      '  process.exit(1);',
      '});',
    ].join('\n');

    execFileSync(process.execPath, ['--input-type=module', '-e', assertionScript], {
      cwd: scratch,
      stdio: 'inherit',
    });
  } finally {
    rmSync(tarballPath, { force: true });
    rmSync(scratch, { recursive: true, force: true });
  }
}

main();
