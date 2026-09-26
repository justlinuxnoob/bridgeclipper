const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')
const vm = require('node:vm')
const ts = require('typescript')
const { fileLinksAvailable } = require('../support/symlinks.cjs')

function transpile(file) {
  const source = fs.readFileSync(path.join(__dirname, '../..', file), 'utf8')
  return ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
}
function loadSource(file, mocks = {}) {
  const module = { exports: {} }
  vm.runInNewContext(transpile(`src/main/${file}`), { module, exports: module.exports, require: (id) => mocks[id] ?? require(id), URL, Set, Map, process, Buffer, console, setTimeout, clearTimeout, __dirname: path.join(__dirname, '../../src/main') })
  return module.exports
}
function loadShared(file) {
  const module = { exports: {} }
  vm.runInNewContext(transpile(`src/shared/${file}`), { module, exports: module.exports, require, URL })
  return module.exports
}

const sharedLogo = loadShared('logo.ts')
const jobContract = loadShared('job-contract.ts')
const videoSource = loadShared('video-source.ts')

/** A minimal PNG: signature, IHDR with the colour type, optional chunks, IDAT and IEND. CRCs are not checked. */
function png({ colorType = 6, width = 4, height = 2, before = [], after = [] } = {}) {
  const chunk = (type, data = Buffer.alloc(0)) => {
    const length = Buffer.alloc(4)
    length.writeUInt32BE(data.length)
    return Buffer.concat([length, Buffer.from(type, 'ascii'), data, Buffer.alloc(4)])
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8
  ihdr[9] = colorType
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    ...before.map((type) => chunk(type, Buffer.from([0, 0]))),
    chunk('IDAT', Buffer.from([1, 2, 3])),
    ...after.map((type) => chunk(type, Buffer.from([0, 0]))),
    chunk('IEND')
  ])
}

function setup(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'bridgeclip-logo-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const userData = path.join(root, 'userData')
  fs.mkdirSync(userData)
  const logo = loadSource('logo.ts', {
    electron: { app: { getPath: (name) => { assert.equal(name, 'userData'); return userData } } },
    '../shared/logo': sharedLogo
  })
  const validation = loadSource('validation.ts', {
    './security': loadSource('security.ts', { electron: {}, '../shared/brand': loadShared('brand.ts') }),
    '../shared/video-source': videoSource,
    '../shared/job-contract': jobContract,
    '../shared/openrouter-models': loadShared('openrouter-models.ts'),
    '../shared/logo': sharedLogo,
    './logo': logo
  })
  const write = (name, bytes) => {
    const file = path.join(root, name)
    fs.writeFileSync(file, bytes)
    return file
  }
  return { root, userData, logo, validation, write }
}

test('transparency comes from an alpha colour type or a tRNS chunk before the image data', (t) => {
  const { logo } = setup(t)
  assert.deepEqual({ ...logo.inspectPng(png({ colorType: 6, width: 300, height: 100 })) }, { width: 300, height: 100, transparent: true })
  assert.equal(logo.inspectPng(png({ colorType: 4 })).transparent, true)
  assert.equal(logo.inspectPng(png({ colorType: 2 })).transparent, false)
  assert.equal(logo.inspectPng(png({ colorType: 0 })).transparent, false)
  assert.equal(logo.inspectPng(png({ colorType: 3, before: ['PLTE'] })).transparent, false)
  assert.equal(logo.inspectPng(png({ colorType: 3, before: ['PLTE', 'tRNS'] })).transparent, true)
  assert.equal(logo.inspectPng(png({ colorType: 2, before: ['tRNS'] })).transparent, true)
  // A tRNS after IDAT is invalid PNG and does not count.
  assert.equal(logo.inspectPng(png({ colorType: 2, after: ['tRNS'] })).transparent, false)
})

test('non-PNG and malformed headers are rejected', (t) => {
  const { logo } = setup(t)
  const good = png()
  const jpegish = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(60)])
  const badIhdr = Buffer.from(good)
  badIhdr.write('IHDX', 12, 'ascii')
  const zeroWidth = png({ width: 0 })
  const badColour = Buffer.from(good)
  badColour[25] = 5
  for (const bytes of [Buffer.alloc(0), good.subarray(0, 20), jpegish, badIhdr, zeroWidth, badColour]) {
    assert.throws(() => logo.inspectPng(bytes), /Choose a PNG image/)
  }
  // A chunk length running past the end stops the scan instead of reading out of bounds.
  const runaway = Buffer.from(png({ colorType: 2, before: ['tEXt'] }))
  runaway.writeUInt32BE(0xffffffff, 33)
  assert.equal(logo.inspectPng(runaway).transparent, false)
})

test('picked files must be transparent PNGs of at most 5 MB and a sane size', (t) => {
  const { logo, write } = setup(t)
  assert.equal(logo.readLogoFile(write('ok.png', png())).info.transparent, true)
  assert.throws(() => logo.readLogoFile(write('opaque.png', png({ colorType: 2 }))), /no transparency/)
  assert.throws(() => logo.readLogoFile(write('huge-dims.png', png({ width: 9000 }))), /at most 8192×8192/)
  const big = Buffer.concat([png(), Buffer.alloc(sharedLogo.LOGO_MAX_BYTES)])
  assert.throws(() => logo.readLogoFile(write('big.png', big)), /5 MB or smaller/)
  assert.throws(() => logo.readLogoFile(write('text.png', 'not an image')), /Choose a PNG image/)
  assert.throws(() => logo.readLogoFile('relative.png'), /Choose a PNG image/)
  assert.throws(() => logo.readLogoFile(path.join(os.tmpdir(), 'missing-logo-file.png')), /Could not read/)
})

test('a picked logo is copied under userData/logos and only those copies are accepted', (t) => {
  const { root, userData, logo, write } = setup(t)
  const source = write('brand.png', png({ width: 200, height: 100 }))
  const imported = logo.importLogo(source)
  assert.equal(path.dirname(imported.path), path.join(userData, 'logos'))
  assert.match(path.basename(imported.path), /^logo-[0-9a-f-]{36}\.png$/)
  assert.deepEqual(fs.readFileSync(imported.path), fs.readFileSync(source))
  assert.equal(imported.width, 200)
  assert.equal(imported.height, 100)
  assert.ok(imported.dataUrl.startsWith('data:image/png;base64,'))
  if (process.platform !== 'win32') assert.equal(fs.statSync(imported.path).mode & 0o777, 0o600)

  assert.equal(logo.isManagedLogoPath(imported.path), true)
  assert.doesNotThrow(() => logo.assertManagedLogo(imported.path))
  assert.equal(logo.loadLogoPreview(imported.path).path, imported.path)
  const outside = write(path.basename(imported.path), png())
  const nested = path.join(userData, 'logos', 'sub', path.basename(imported.path))
  for (const candidate of [source, outside, nested, path.join(userData, 'logos', 'brand.png'), `${userData}/logos/../logos/${path.basename(imported.path)}`, '', null, 42]) {
    assert.equal(logo.isManagedLogoPath(candidate), false, String(candidate))
    assert.throws(() => logo.assertManagedLogo(candidate), /Choose the logo again/)
  }
  assert.equal(logo.loadLogoPreview(''), null)
  assert.equal(logo.loadLogoPreview(source), null)

  // A managed name whose bytes were later replaced with an opaque image still fails the check.
  fs.writeFileSync(imported.path, png({ colorType: 2 }))
  assert.throws(() => logo.assertManagedLogo(imported.path), /no transparency/)
  assert.equal(logo.loadLogoPreview(imported.path), null)
  assert.ok(root)
})

test('a symlink in the logos folder is not a managed logo', (t) => {
  if (!fileLinksAvailable) return t.skip('File links unavailable')
  const { userData, logo, write } = setup(t)
  const real = logo.importLogo(write('brand.png', png()))
  const link = path.join(userData, 'logos', 'logo-00000000-0000-4000-8000-000000000000.png')
  fs.symlinkSync(write('elsewhere.png', png()), link)
  assert.equal(logo.isManagedLogoPath(link), false)
  assert.equal(logo.isManagedLogoPath(real.path), true)
})

test('pruning keeps the saved logo and logos used by queued jobs', (t) => {
  const { logo, write, userData } = setup(t)
  const source = write('brand.png', png())
  const [a, b, c] = [logo.importLogo(source), logo.importLogo(source), logo.importLogo(source)]
  const stray = path.join(userData, 'logos', 'notes.txt')
  fs.writeFileSync(stray, 'keep me')
  logo.pruneLogos([a.path, undefined, b.path, null])
  assert.equal(fs.existsSync(a.path), true)
  assert.equal(fs.existsSync(b.path), true)
  assert.equal(fs.existsSync(c.path), false)
  assert.equal(fs.existsSync(stray), true, 'only files with our generated names are removed')
})

test('job validation accepts an in-range logo on the managed copy and rejects everything else', (t) => {
  const { logo, validation, write } = setup(t)
  const managed = logo.importLogo(write('brand.png', png())).path
  const job = { videoUrl: 'https://example.com/video', maxClips: 5, autoClipCount: true, includeCaptions: true, aspectRatio: '9:16', layoutStyle: 'auto', layoutVision: true, pacing: 'tight', captionPreset: 'pop', durationRanges: ['short'], startTimeSeconds: null, endTimeSeconds: null, bannerPlatform: null, bannerChannelUrl: null }
  const good = { path: managed, x: 0.81, y: 0.0225, width: 0.15, opacity: 0.8 }

  assert.equal(validation.validateJobConfig(job).logo, undefined)
  assert.equal(validation.validateJobConfig({ ...job, logo: null }).logo, undefined)
  assert.deepEqual({ ...validation.validateJobConfig({ ...job, logo: good }).logo }, good)
  const withoutOpacity = { path: good.path, x: good.x, y: good.y, width: good.width }
  assert.deepEqual({ ...validation.validateJobConfig({ ...job, logo: withoutOpacity }).logo }, withoutOpacity)
  assert.doesNotThrow(() => validation.validateJobConfig({ ...job, logo: { ...good, x: 0, y: 0, width: 0.5 } }))
  assert.doesNotThrow(() => validation.validateJobConfig({ ...job, logo: { ...good, x: 0.5, y: 1, width: 0.5 } }))

  const invalid = [
    'logo', 1, [], true,
    { ...good, x: -0.01 }, { ...good, x: 1.01 }, { ...good, y: -1 }, { ...good, y: 1.5 },
    { ...good, x: NaN }, { ...good, y: Infinity }, { ...good, x: '0.5' },
    { ...good, width: 0 }, { ...good, width: 0.04 }, { ...good, width: 0.51 }, { ...good, width: null },
    { ...good, x: 0.9, width: 0.15 },
    { ...good, opacity: 0.1 }, { ...good, opacity: 1.2 }, { ...good, opacity: '1' },
    { ...good, extra: true },
    { ...good, path: write('brand-copy.png', png()) },
    { ...good, path: '/etc/passwd' },
    { ...good, path: 'logo.png' },
    { x: 0.1, y: 0.1, width: 0.2 }
  ]
  for (const bad of invalid) assert.throws(() => validation.validateJobConfig({ ...job, logo: bad }), undefined, JSON.stringify(bad))
})

test('the bridge request carries the logo only when one is set', () => {
  const { PassThrough } = require('node:stream')
  const { EventEmitter } = require('node:events')
  const forward = (config) => {
    const child = new EventEmitter()
    child.stdin = new PassThrough()
    child.stdout = new PassThrough()
    child.stderr = new PassThrough()
    let input = ''
    child.stdin.on('data', (chunk) => { input += chunk.toString() })
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'bridgeclip-logo-runner-'))
    try {
      const runner = loadSource('pipeline-runner.ts', {
        electron: { app: { isPackaged: false, getPath: () => home } },
        fs: { ...fs, existsSync: () => true },
        child_process: { execFile: require('node:child_process').execFile, spawn: () => child },
        './settings-store': { loadSettings: () => ({ outputDirectory: '/tmp', pythonPath: 'python3' }), getSettingsForBridge: () => ({}), vocabularyTerms: () => [] },
        './logger': { logger: { info() {}, error() {}, warn() {} } },
        '../shared/job-output': loadShared('job-output.ts'),
        './run-history': { finishRunRecord() {} },
        '../shared/job-contract': jobContract,
        './tools': { resolveBinary: (name) => name }
      })
      runner.startClipJob('logo-job', config, { isDestroyed: () => false, webContents: { isDestroyed: () => false, send() {} } })
      return JSON.parse(input)
    } finally { fs.rmSync(home, { recursive: true, force: true }) }
  }
  const base = { videoUrl: '/tmp/video.mp4', captionPreset: 'pop', includeCaptions: true }
  const without = forward(base)
  assert.equal('logo' in without, false)
  assert.deepEqual(forward({ ...base, logo: undefined }), without)
  assert.deepEqual(forward({ ...base, logo: null }), without)
  const withLogo = forward({ ...base, logo: { path: '/data/logos/logo.png', x: 0.1, y: 0.2, width: 0.3 } })
  assert.deepEqual(withLogo.logo, { path: '/data/logos/logo.png', x: 0.1, y: 0.2, width: 0.3, opacity: 1 })
  delete withLogo.logo
  assert.deepEqual(withLogo, without)
})

test('placement defaults to the top-right corner and is clamped inside the frame', () => {
  const { defaultLogoPlacement, clampLogoPlacement, overlapsPlatformUi, logoHeightFraction } = sharedLogo
  const placement = defaultLogoPlacement(0.5)
  assert.ok(Math.abs(placement.x - 0.81) < 1e-9)
  assert.ok(Math.abs(placement.y - 0.04 * 9 / 16) < 1e-9)
  assert.equal(placement.width, 0.15)
  assert.equal(overlapsPlatformUi(placement, 0.5), false, 'the default does not trigger the platform warning')

  assert.deepEqual({ ...clampLogoPlacement({ x: 0.95, y: 0.99, width: 0.3 }, 1) }, { x: 0.7, y: 1 - logoHeightFraction(0.3, 1), width: 0.3 })
  assert.deepEqual({ ...clampLogoPlacement({ x: -1, y: -1, width: 2 }, 1) }, { x: 0, y: 0, width: 0.5 })
  assert.equal(clampLogoPlacement({ x: 0, y: 0, width: 0 }, 1).width, 0.05)
  assert.equal(clampLogoPlacement({ x: NaN, y: 0, width: 0.2 }, 1).x, 0)
  // A tall logo on a 16:9 frame shrinks so its height fits.
  const tall = clampLogoPlacement({ x: 0, y: 0, width: 0.5 }, 10, 16 / 9)
  assert.ok(logoHeightFraction(tall.width, 10, 16 / 9) <= 1 + 1e-9)

  assert.equal(overlapsPlatformUi({ x: 0.1, y: 0.85, width: 0.15 }, 0.5), true, 'bottom caption area')
  assert.equal(overlapsPlatformUi({ x: 0.84, y: 0.5, width: 0.15 }, 0.5), true, 'right-hand buttons')
  assert.equal(overlapsPlatformUi({ x: 0.04, y: 0.02, width: 0.15 }, 0.5), false)
})
