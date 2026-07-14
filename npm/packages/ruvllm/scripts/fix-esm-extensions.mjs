#!/usr/bin/env node
/**
 * Post-build fixer for the ESM output of this package (#656).
 *
 * `tsconfig.esm.json` compiles with `"module": "ESNext"` +
 * `"moduleResolution": "Node"` (classic resolution). In that mode
 * TypeScript does not rewrite relative import/export specifiers to
 * include a file extension, so the emitted `dist/esm/*.js` files contain
 * specifiers like `from './types'`. Node's native ESM resolver requires an
 * explicit extension for relative specifiers, so `import('@ruvector/ruvllm')`
 * from a `"type": "module"` consumer throws `ERR_MODULE_NOT_FOUND`.
 *
 * This script walks the compiled `dist/esm/` tree after `tsc` runs and
 * rewrites every extensionless relative specifier to whatever actually
 * exists on disk (`<spec>.js` or `<spec>/index.js`), in both `.js` and
 * `.d.ts` files — declaration files need the same `.js`-suffixed
 * specifiers (the TypeScript NodeNext convention) even though they only
 * carry types, so that consumers' type-checkers resolve them too.
 *
 * Zero external dependencies: only `node:fs`, `node:path`, `node:url`.
 *
 * Usage:
 *   node scripts/fix-esm-extensions.mjs            # rewrite in place
 *   node scripts/fix-esm-extensions.mjs --verify    # assert-only, no writes
 */

import { readdirSync, statSync, readFileSync, writeFileSync } from 'node:fs';
import { join, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST_ESM = join(__dirname, '..', 'dist', 'esm');
const VERIFY = process.argv.includes('--verify');

// Matches the specifier in `... from '<spec>'` / `... from "<spec>"` —
// covers `import X from`, `import { a } from`, `export { a } from`,
// `export * from`, `export * as ns from`.
const FROM_SPEC_RE = /\bfrom(\s*)(['"])(\.[^'"]+)\2/g;

// Matches a side-effect import: `import '<spec>';` — the bare form with no
// bindings and no `from` clause, anchored to the start of a line.
const SIDE_EFFECT_SPEC_RE = /^(\s*)import(\s*)(['"])(\.[^'"]+)\3/gm;

function walk(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, out);
    } else if (entry.isFile() && (full.endsWith('.js') || full.endsWith('.d.ts'))) {
      out.push(full);
    }
  }
  return out;
}

function existsFile(p) {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

// Only extensionless relative specifiers are our concern; anything already
// carrying an extension (e.g. already-fixed `.js`, or a hypothetical
// `.json`) is left untouched — this is what makes the script idempotent.
function needsFix(spec) {
  return (spec.startsWith('./') || spec.startsWith('../')) && extname(spec) === '';
}

// Resolve an extensionless relative specifier against the directory of the
// file that references it, against what actually exists in the compiled
// dist/esm tree. Returns the rewritten specifier, or null if neither the
// `<spec>.js` file nor the `<spec>/index.js` barrel exists on disk.
function resolveSpecifier(fileDir, spec) {
  const asFile = join(fileDir, `${spec}.js`);
  if (existsFile(asFile)) return `${spec}.js`;
  const asIndex = join(fileDir, spec, 'index.js');
  if (existsFile(asIndex)) return `${spec}/index.js`;
  return null;
}

// Scans `text` for extensionless relative specifiers via both patterns.
// `onSpec(spec)` is called for each one found; if it returns a string, that
// string replaces the specifier's quoted body in the output. Returns
// { text, violations } where `violations` lists specifiers `onSpec`
// declined to rewrite (returned a falsy value).
function scanAndRewrite(source, onSpec) {
  const violations = [];

  const rewriteFrom = (fullMatch, ws, quote, spec) => {
    if (!needsFix(spec)) return fullMatch;
    const replacement = onSpec(spec);
    if (!replacement) {
      violations.push(spec);
      return fullMatch;
    }
    return `from${ws}${quote}${replacement}${quote}`;
  };

  const rewriteSideEffect = (fullMatch, lead, ws, quote, spec) => {
    if (!needsFix(spec)) return fullMatch;
    const replacement = onSpec(spec);
    if (!replacement) {
      violations.push(spec);
      return fullMatch;
    }
    return `${lead}import${ws}${quote}${replacement}${quote}`;
  };

  let text = source.replace(FROM_SPEC_RE, rewriteFrom);
  text = text.replace(SIDE_EFFECT_SPEC_RE, rewriteSideEffect);

  return { text, violations };
}

function fixFile(file) {
  const dir = dirname(file);
  const original = readFileSync(file, 'utf8');

  const { text, violations } = scanAndRewrite(original, (spec) => resolveSpecifier(dir, spec));

  if (violations.length > 0) {
    for (const spec of violations) {
      console.error(
        `fix-esm-extensions: cannot resolve relative specifier '${spec}' referenced from ${file} ` +
          `(looked for '${spec}.js' and '${spec}/index.js' in ${dir})`
      );
    }
    process.exit(1);
  }

  if (text !== original) {
    writeFileSync(file, text, 'utf8');
    return true;
  }
  return false;
}

function verifyFile(file) {
  const remaining = [];
  // In verify mode we don't need resolution at all: any extensionless
  // relative specifier still present after the fix step is itself the
  // failure condition.
  scanAndRewrite(readFileSync(file, 'utf8'), (spec) => {
    remaining.push(spec);
    return null; // never rewrite in verify mode; force it into `violations`
  });
  return remaining.map((spec) => ({ file, spec }));
}

function main() {
  let files;
  try {
    files = walk(DIST_ESM);
  } catch (err) {
    console.error(`fix-esm-extensions: cannot read ${DIST_ESM}: ${err.message}`);
    process.exit(1);
  }

  if (VERIFY) {
    const remaining = files.flatMap(verifyFile);
    if (remaining.length > 0) {
      for (const { file, spec } of remaining) {
        console.error(`fix-esm-extensions --verify: extensionless relative specifier '${spec}' in ${file}`);
      }
      console.error(`fix-esm-extensions --verify: FAILED — ${remaining.length} extensionless specifier(s) remain`);
      process.exit(1);
    }
    console.log(`fix-esm-extensions --verify: OK — 0 extensionless relative specifiers across ${files.length} files`);
    return;
  }

  let changed = 0;
  for (const file of files) {
    if (fixFile(file)) changed += 1;
  }
  console.log(`fix-esm-extensions: rewrote ${changed}/${files.length} file(s) under dist/esm`);
}

main();
