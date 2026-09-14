import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/playback.ts', import.meta.url), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { PlaybackClock } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

function simulate(seconds, duration = 120000) {
  const clock = new PlaybackClock();
  let queue = 0, reserved = 0, finish = null, busy = 0, frames = 0;
  for (let now = 0; now < duration; now += 1000 / 60) {
    if (finish !== null && now >= finish) {
      clock.observe(now, 12, queue);
      queue += 12;
      finish = null;
    }
    if (queue && clock.ready(now)) { queue--; reserved--; frames++; }
    if (finish === null && reserved + 12 <= 24) {
      reserved += 12;
      finish = now + seconds * 1000;
    }
    if (finish !== null) busy += 1000 / 60;
    assert.ok(reserved <= 24);
  }
  return { fps: frames / (duration / 1000), busy: busy / duration };
}

test('paced output keeps a bounded generator busy at different speeds', () => {
  for (const seconds of [.6, .85, 1, 1.6, 2.4]) {
    const result = simulate(seconds);
    assert.ok(result.busy > .97, JSON.stringify({ seconds, ...result }));
    assert.ok(result.fps > (12 / seconds) * .92, JSON.stringify({ seconds, ...result }));
  }
});

test('slow arrivals are supported below 10 fps', () => {
  const clock = new PlaybackClock();
  for (let i = 0; i < 40; i++) clock.observe(i * 6000, 12, 0);
  assert.ok(clock.interval > 490);
});

test('an empty-queue stall does not cause an immediate burst', () => {
  const clock = new PlaybackClock();
  assert.equal(clock.ready(10000), true);
  assert.equal(clock.ready(10000), false);
  assert.equal(clock.ready(10001), false);
});

test('more buffered output speeds up delivery', () => {
  const normal = new PlaybackClock(), buffered = new PlaybackClock();
  for (let i = 0; i < 20; i++) {
    normal.observe(i * 1200, 12, 0);
    buffered.observe(i * 1200, 12, 12);
  }
  assert.ok(buffered.interval < normal.interval * .6);
});
