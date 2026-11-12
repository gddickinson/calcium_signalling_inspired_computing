// Helpers for the node engine tests (no tests of its own). Fixtures live in tests/fixtures/<name>.json, made by
// scripts/make_<name>_fixture.py from the repo's Python model classes after np.random.seed.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

export const fixture = name =>
  JSON.parse(readFileSync(new URL(`../../tests/fixtures/${name}.json`, import.meta.url), 'utf8'));

const num = v => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v === 'nan' ? NaN : v);

// Deep comparison with a relative tolerance on numbers ("inf"/"-inf" strings in the fixture mean infinities).
export function near(actual, expected, path = '$', rtol = 1e-9) {
  expected = num(expected);
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${path}: expected an array`);
    assert.equal(actual.length, expected.length, `${path}: length`);
    expected.forEach((e, i) => near(actual[i], e, `${path}[${i}]`, rtol));
  } else if (expected && typeof expected === 'object') {
    for (const k of Object.keys(expected)) near(actual[k], expected[k], `${path}.${k}`, rtol);
  } else if (typeof expected === 'number') {
    if (!Number.isFinite(expected)) { assert.ok(Object.is(actual, expected) || (Number.isNaN(expected) && Number.isNaN(actual)), `${path}: ${actual} != ${expected}`); return; }
    assert.ok(Math.abs(actual - expected) <= 1e-12 + rtol * Math.max(Math.abs(actual), Math.abs(expected)), `${path}: ${actual} != ${expected}`);
  } else assert.equal(actual, expected, path);
}
