const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/static/app.js', 'utf8');

function form() {
  const fields = new Map();
  const $ = (id) => {
    if (!fields.has(id)) fields.set(id, {
      value: '', handlers: {}, children: [], classList: { toggle() {} },
      append(...nodes) { this.children.push(...nodes); },
      addEventListener(event, handler) { this.handlers[event] = handler; },
      replaceChildren(...options) { this.children = options; this.value = options[0]?.value || ''; },
      reportValidity: () => true,
    });
    return fields.get(id);
  };
  const notices = [];
  const context = vm.createContext({ $, Option: function (text, value) { this.value = String(value); },
    state: { jobs: [], active: 'video' }, notify: (text) => notices.push(text), loadVersion: 1,
    resetClips() {}, renderHighlightState() {}, document: { createElement: (tag) => ({
      children: [], append(...nodes) { this.children.push(...nodes); }, textContent: '',
      addEventListener() {},
    }) },
  });
  vm.runInContext(source.slice(source.indexOf('let llmSettings = null;'),
    source.indexOf('function renderHighlightState(job)')), context);
  const submitStart = source.indexOf('$("highlights-form").addEventListener("submit"');
  vm.runInContext(source.slice(submitStart, source.indexOf('\nrefreshLLMSettings();', submitStart)), context);
  vm.runInContext(`
    llmSettings = {llm: {configured: true}, highlights: {max_count: 5, min_seconds: 20, max_seconds: 90, objective: 'General selection.'}};
    promptCatalog = [
      {id: 'acquisition', area: 'acquisition', name: 'Adquisición', status: 'master', revisions: [
        {version: 1, min_seconds: 15, max_seconds: 45, objective: 'Acquisition selection.'},
        {version: 2, min_seconds: 10, max_seconds: 60, objective: 'Updated acquisition selection.'}
      ]},
      {id: 'activation', area: 'activation', name: 'Activación', status: 'master', revisions: [
        {version: 1, min_seconds: 30, max_seconds: 90, objective: 'Activation selection.'}
      ]}
    ];
  `, context);
  $('clip-area').value = 'acquisition';
  $('audience-context').value = 'Docentes de primaria';
  $('highlight-count').value = '5';
  $('clip-area').handlers.change();
  return { $, context, notices };
}

function variant(changes = {}) {
  return {id: 'draft', version: 1, area: 'acquisition', audience: 'Docentes de primaria',
    min_seconds: 10, max_seconds: 60, master: {id: 'acquisition', version: 2, objective: 'Updated acquisition selection.'},
    objective: 'Personalized teacher examples.', prior_learning: '', channel: '', changes: ['Use examples.'],
    assumptions: ['No causal evidence.'], evidence: [], reviewed_at: null, ...changes};
}

test('area applies its master and duration without a need selector', () => {
  const { $ } = form();
  assert.equal($('highlight-objective').value, 'Updated acquisition selection.');
  assert.equal($('highlight-min').value, 10);
  $('clip-area').value = 'activation';
  $('clip-area').handlers.change();
  assert.equal($('highlight-objective').value, 'Activation selection.');
  assert.equal($('highlight-min').value, 30);
  const html = fs.readFileSync('app/static/index.html', 'utf8');
  assert.doesNotMatch(html, /id="clip-need"|id="audience-kind"/);
  assert.ok(html.indexOf('id="prompt-manager"') < html.indexOf('id="highlight-objective"'));
});

test('an outdated server explains why the masters cannot load', async () => {
  const { $, context } = form();
  const settings = vm.runInContext('llmSettings', context);
  context.api = async (path) => { if (path === '/api/settings') return settings; throw new Error('Not Found'); };
  await context.refreshLLMSettings();
  assert.match($('prompt-status').textContent, /Reinicia el servidor y recarga/);
});

test('an area selected before catalog loading is retained', async () => {
  const { $, context } = form();
  const catalog = vm.runInContext('promptCatalog', context), settings = vm.runInContext('llmSettings', context);
  vm.runInContext('promptCatalog = [];', context);
  context.api = async (path) => path === '/api/settings' ? settings : catalog;
  $('clip-area').value = 'activation'; $('clip-area').handlers.change();
  await context.refreshLLMSettings();
  assert.equal($('highlight-objective').value, 'Activation selection.');
  assert.equal($('clip-area').value, 'activation');
});

test('changing audience invalidates an approved variant', () => {
  const { $, context } = form();
  context.approved = variant({reviewed_at: 'today'});
  vm.runInContext('showVariant(approved);', context);
  $('audience-context').value = 'Otra audiencia'; $('audience-context').handlers.input();
  assert.equal(vm.runInContext('promptVariant', context), null);
  assert.equal($('highlight-objective').value, 'Updated acquisition selection.');
});

test('an unreviewed proposal or unsaved edit cannot start clip selection', async () => {
  const { $, context, notices } = form();
  context.api = () => { throw new Error('Selection must not be submitted'); };
  context.draft = variant(); vm.runInContext('showVariant(draft);', context);
  await $('highlights-form').handlers.submit({preventDefault() {}});
  assert.match(notices[0], /Revisa y guarda/);
  assert.equal($('prompt-manager').open, true);
});

test('approved prompt, audience and custom durations are submitted together', async () => {
  const { $, context } = form();
  context.approved = variant({reviewed_at: 'today'});
  vm.runInContext('showVariant(approved);', context);
  const calls = [];
  context.api = async (path, options) => { calls.push(JSON.parse(options.body)); return {id: 'video'}; };
  await $('highlights-form').handlers.submit({preventDefault() {}});
  assert.equal(calls[0].personalization_id, 'draft');
  assert.equal(calls[0].audience_context, 'Docentes de primaria');
  assert.equal(calls[0].min_seconds, 10);
});

test('a delayed proposal cannot replace a changed audience', async () => {
  const { $, context } = form();
  let finish;
  context.api = () => new Promise((resolve) => { finish = resolve; });
  const pending = $('adapt-prompt').handlers.click();
  $('audience-context').value = 'Otra audiencia'; $('audience-context').handlers.input();
  finish(variant()); await pending;
  assert.equal(vm.runInContext('promptVariant', context), null);
  assert.equal($('highlight-objective').value, 'Updated acquisition selection.');
});

test('restoring a job restores its reviewed variant and original context', () => {
  const { $, context } = form();
  context.approved = variant({reviewed_at: 'today', channel: 'Instagram', prior_learning: 'Examples worked.'});
  context.state.jobs = [{id: 'video', highlights_prompt: {personalization: context.approved}}];
  vm.runInContext(`fillHighlightOptions({prompt_id: 'acquisition', prompt_version: 2, personalization_id: 'draft',
    max_count: 3, min_seconds: 10, max_seconds: 60, audience_context: 'Docentes de primaria'});`, context);
  assert.equal($('highlight-objective').value, 'Personalized teacher examples.');
  assert.equal($('audience-channel').value, 'Instagram');
  assert.equal($('prior-learning').value, 'Examples worked.');
});

test('a saved personalization survives restoring its job before the catalog arrives', async () => {
  const { $, context } = form();
  const catalog = vm.runInContext('promptCatalog', context), settings = vm.runInContext('llmSettings', context);
  context.approved = variant({reviewed_at: 'today'});
  context.state.jobs = [{id: 'video', highlights_prompt: {personalization: context.approved}}];
  context.api = async (path) => path === '/api/settings' ? settings : catalog;
  vm.runInContext(`promptCatalog = []; fillHighlightOptions({prompt_id: 'acquisition', prompt_version: 2,
    personalization_id: 'draft', max_count: 3, min_seconds: 10, max_seconds: 60,
    audience_context: 'Docentes de primaria'});`, context);
  await context.refreshLLMSettings();
  assert.equal($('highlight-objective').value, 'Personalized teacher examples.');
  assert.equal(vm.runInContext('promptVariant.id', context), 'draft');
});

test('campaign changes invalidate reviewed personalizations and are submitted explicitly', async () => {
  const { $, context } = form();
  context.approved = variant({reviewed_at: 'today'});
  vm.runInContext('showVariant(approved);', context);
  $('campaign-context').value = 'Definir objetivos antes de decidir';
  $('campaign-context').handlers.input();
  assert.equal(vm.runInContext('promptVariant', context), null);
  const calls = [];
  context.api = async (path, options) => { calls.push(JSON.parse(options.body)); return {id: 'video'}; };
  await $('highlights-form').handlers.submit({preventDefault() {}});
  assert.equal(calls[0].campaign_context, 'Definir objetivos antes de decidir');
});

test('ordinary regeneration uses the latest master without changing historical results', () => {
  const { $, context } = form();
  vm.runInContext(`fillHighlightOptions({prompt_id: 'acquisition', prompt_version: 1,
    max_count: 5, min_seconds: 15, max_seconds: 45, audience_context: ''});`, context);
  assert.equal(Number($('prompt-version').value), 2);
  assert.equal($('highlight-objective').value, 'Updated acquisition selection.');
});
