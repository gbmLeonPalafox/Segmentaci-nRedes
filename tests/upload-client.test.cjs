const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function uploader(fetch) {
  const source = fs.readFileSync('app/static/app.js', 'utf8');
  const code = source.slice(source.indexOf('async function uploadFile('), source.indexOf('$("import-form").addEventListener'));
  const span = { textContent: '' };
  const context = vm.createContext({
    api: async () => ({ id: 'session', chunk_bytes: 4 }), fetch,
    $: () => ({ querySelector: () => span }), AbortSignal,
    setTimeout: (callback) => callback(),
  });
  vm.runInContext(code, context);
  return { upload: context.uploadFile, span };
}
const file = { name: 'video.mp4', size: 9, slice: (start, end) => ({ start, end }) };
const ok = (body) => ({ ok: true, json: async () => body });

test('retries a lost chunk acknowledgement and completion response without skipping bytes', async () => {
  const offsets = [];
  let completionCalls = 0;
  const { upload, span } = uploader(async (path, options) => {
    assert.equal(options.signal.aborted, false);
    if (path.endsWith('/complete')) {
      if (++completionCalls === 1) throw new TypeError('Connection reset after completion');
      return ok({ id: 'job' });
    }
    const offset = Number(new URL(path, 'http://localhost').searchParams.get('offset'));
    offsets.push(offset);
    if (offsets.length === 1) throw new TypeError('Lost acknowledgement');
    assert.equal(options.body.start, offset);
    return ok({ offset: options.body.end });
  });
  assert.equal((await upload(file, 'whisper-1', 'es')).id, 'job');
  assert.deepEqual(offsets, [0, 0, 4, 8]);
  assert.equal(completionCalls, 2);
  assert.match(span.textContent, /Upload complete/);
});

test('rejects a permanent error and discards the unfinished session', async () => {
  const methods = [];
  const { upload } = uploader(async (path, options) => {
    methods.push(options.method);
    if (options.method === 'DELETE') return ok({});
    return { ok: false, status: 413, json: async () => ({ detail: 'Too large' }) };
  });
  await assert.rejects(upload(file, 'whisper-1', ''), /Too large/);
  assert.deepEqual(methods, ['PUT', 'DELETE']);
});

test('stops after three retries when the connection stays unavailable', async () => {
  let attempts = 0;
  const { upload } = uploader(async (path, options) => {
    if (options.method === 'DELETE') return ok({});
    attempts++;
    throw new TypeError('Offline');
  });
  await assert.rejects(upload(file, 'whisper-1', ''), /Offline/);
  assert.equal(attempts, 4);
});
