/**
 * Luoys APK native-emulator integration test — real-world APK end-to-end test.
 *
 * Tests the complete workflow:
 * 1. Extract arm64-v8a libs from luoys-6.10.apk
 * 2. Create emulator session
 * 3. Load third-party .so (skip libapp.so/libflutter.so — Dart runtime only)
 * 4. Inspect imports (diagnose unresolved)
 * 5. List exported symbols
 * 6. Call exported functions / JNI exports
 * 7. Trace execution
 * 8. Detect ISA/JNI gaps and report
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { NativeEmulatorHandlers } from '@server/domains/native-emulator/handlers.impl';
import { extractArm64Libs } from '@modules/native-emulator/apk';

const APK_PATH = 'D:/cumhub/reverse/luolishe/luoys-6.10.apk';

/** Parse the JSON payload out of an MCP text response (same as handlers.test.ts). */
// biome-ignore lint: any required for generic JSON deserialization
function payload(res: any): any {
  if (typeof res === 'string') return JSON.parse(res);
  if (res?.content?.[0]?.text) return JSON.parse(res.content[0].text);
  return res;
}

// Skip if APK not present (CI environment)
const APK_EXISTS = await (async () => {
  try {
    const { existsSync } = await import('node:fs');
    return existsSync(APK_PATH);
  } catch {
    return false;
  }
})();

describe.skipIf(!APK_EXISTS)('luoys APK integration test', () => {
  let handlers: NativeEmulatorHandlers;
  let sessionId: string;
  let extractedLibs: Awaited<ReturnType<typeof extractArm64Libs>>;

  beforeAll(async () => {
    handlers = new NativeEmulatorHandlers();
    // Extract all arm64-v8a libs
    extractedLibs = await extractArm64Libs(APK_PATH);
    console.log(
      `Extracted ${extractedLibs.length} libs:`,
      extractedLibs.map((l) => l.name),
    );
  });

  afterAll(async () => {
    if (sessionId) {
      await handlers.handleDestroySession({ sessionId });
    }
    handlers.dispose();
  });

  it('extracts arm64-v8a libraries from luoys APK', () => {
    expect(extractedLibs.length).toBeGreaterThan(0);
    // Should have libsqlite3, libmmkv, libijkplayer, libflutter, libapp
    const names = extractedLibs.map((l) => l.name);
    expect(names).toContain('libsqlite3.so');
    expect(names).toContain('libmmkv.so');
    expect(names).toContain('libflutter.so');
    expect(names).toContain('libapp.so');
  });

  it('creates an emulator session', async () => {
    const result = await handlers.handleCreateSession({});
    const data = payload(result);
    sessionId = data.sessionId as string;
    expect(sessionId).toBeTruthy();
    console.log(`Created session: ${sessionId}`);
  });

  it('loads libsqlite3.so and inspects symbols', async () => {
    expect(sessionId).toBeTruthy();

    // Write libsqlite3.so to temp file
    const { writeFile, mkdtemp, rm } = await import('node:fs/promises');
    const { join } = await import('node:path');
    const { tmpdir } = await import('node:os');

    const tmpDir = await mkdtemp(join(tmpdir(), 'luoys-test-'));
    const soPath = join(tmpDir, 'libsqlite3.so');
    const sqlite3Lib = extractedLibs.find((l) => l.name === 'libsqlite3.so');
    expect(sqlite3Lib).toBeTruthy();

    await writeFile(soPath, sqlite3Lib!.bytes);

    try {
      // Inspect imports first
      const inspectResult = await handlers.handleInspectImports({ soPath });
      const inspectData = payload(inspectResult);
      console.log(
        `Imports: ${inspectData.unresolvedCount} unresolved, ${inspectData.bionicResolvedCount} bionic-resolved`,
      );
      if (inspectData.unresolvedImports?.length > 0) {
        console.log('Unresolved:', inspectData.unresolvedImports.slice(0, 5));
      }

      // Load library
      const loadResult = await handlers.handleLoadLibrary({ sessionId, soPath });
      const loadData = payload(loadResult);
      console.log(
        `Loaded: entry=0x${loadData.entry?.toString(16)}, unresolved=${loadData.unresolvedImports?.length ?? 0}`,
      );

      // List symbols
      const symbolsResult = await handlers.handleListSymbols({ sessionId });
      const symbolsData = payload(symbolsResult);
      console.log(`Symbols: ${symbolsData.symbols?.length ?? 0} exports`);

      // Find sqlite3_version or sqlite3_initialize
      const exports = symbolsData.symbols as string[];
      const versionSym = exports.find((s) => s === 'sqlite3_libversion');
      const initSym = exports.find((s) => s === 'sqlite3_initialize');

      console.log('Key symbols:', {
        sqlite3_libversion: versionSym ? 'found' : 'not found',
        sqlite3_initialize: initSym ? 'found' : 'not found',
      });

      // Try calling sqlite3_initialize if present
      if (initSym) {
        const callResult = await handlers.handleCallSymbol({
          sessionId,
          symbol: 'sqlite3_initialize',
          args: [],
        });

        const callData = payload(callResult);
        if (callData.success === false || callData.error) {
          const errorMsg = callData.error || callData.message || 'unknown error';
          console.log(`sqlite3_initialize failed: ${errorMsg}`);
          // Check if it's an ISA gap
          if (
            String(errorMsg).includes('Unsupported opcode') ||
            String(errorMsg).includes('unimplemented')
          ) {
            console.log('⚠️  ISA gap detected!');
          }
        } else {
          console.log(`sqlite3_initialize returned: ${callData.result}`);
        }
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('loads libmmkv.so and tests JNI exports', async () => {
    // Create a NEW session for libmmkv.so (each session can only load one .so)
    const sessionResult = await handlers.handleCreateSession({});
    const sessionData = payload(sessionResult);
    const mmkvSessionId = sessionData.sessionId as string;
    expect(mmkvSessionId).toBeTruthy();

    const { writeFile, mkdtemp, rm } = await import('node:fs/promises');
    const { join } = await import('node:path');
    const { tmpdir } = await import('node:os');

    const tmpDir = await mkdtemp(join(tmpdir(), 'luoys-mmkv-'));
    const soPath = join(tmpDir, 'libmmkv.so');
    const mmkvLib = extractedLibs.find((l) => l.name === 'libmmkv.so');
    expect(mmkvLib).toBeTruthy();

    await writeFile(soPath, mmkvLib!.bytes);

    try {
      // Load library into the NEW session
      const loadResult = await handlers.handleLoadLibrary({ sessionId: mmkvSessionId, soPath });
      const loadData = payload(loadResult);

      // Log the error if load failed
      if (loadData.success === false || loadData.error) {
        console.log(
          '❌ libmmkv.so load failed:',
          loadData.error || loadData.message || JSON.stringify(loadData),
        );
      } else {
        console.log('✅ libmmkv.so loaded successfully');
        console.log(`  Constructor faults: ${loadData.constructorFaults?.length ?? 0}`);
      }

      expect(loadData.success !== false).toBe(true);

      // List symbols
      const symbolsResult = await handlers.handleListSymbols({ sessionId: mmkvSessionId });
      const symbolsData = payload(symbolsResult);
      const exports = symbolsData.symbols as string[]; // symbols is an array of strings, not objects

      // Find JNI exports
      const jniExports = exports.filter((s) => s.startsWith('Java_'));
      console.log(`Found ${jniExports.length} JNI exports`);
      if (jniExports.length > 0) {
        console.log('Sample JNI exports:', jniExports.slice(0, 3));
      }

      // Try calling a JNI export if present
      if (jniExports.length > 0) {
        const target = jniExports[0];
        console.log(`Attempting to call: ${target}`);

        const callResult = await handlers.handleCallJniExport({
          sessionId: mmkvSessionId,
          symbol: target,
          javaArgs: [],
        });

        const callData = payload(callResult);
        if (callData.success === false || callData.error) {
          const errorMsg = callData.error || callData.message || 'unknown error';
          console.log(`${target} failed: ${errorMsg}`);
          if (String(errorMsg).includes('Unsupported opcode') || String(errorMsg).includes('JNI')) {
            console.log('⚠️  ISA/JNI gap detected!');
          }
        } else {
          console.log(`${target} returned: ${callData.result}`);
        }
      }

      // Clean up mmkv session
      await handlers.handleDestroySession({ sessionId: mmkvSessionId });
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('traces execution on a small library', async () => {
    expect(sessionId).toBeTruthy();

    const { writeFile, mkdtemp, rm } = await import('node:fs/promises');
    const { join } = await import('node:path');
    const { tmpdir } = await import('node:os');

    // Use the smallest lib for trace test
    const smallLib =
      extractedLibs.find((l) => l.name === 'libsurface_util_jni.so') ||
      extractedLibs.find((l) => l.name === 'libimage_processing_util_jni.so');

    if (!smallLib) {
      console.log('No small lib found for trace test, skipping');
      return;
    }

    const tmpDir = await mkdtemp(join(tmpdir(), 'luoys-trace-'));
    const soPath = join(tmpDir, smallLib.name);

    await writeFile(soPath, smallLib.bytes);

    try {
      const loadResult = await handlers.handleLoadLibrary({ sessionId, soPath });
      const loadData = payload(loadResult);
      expect(loadData.success !== false).toBe(true);

      const symbolsResult = await handlers.handleListSymbols({ sessionId });
      const symbolsData = payload(symbolsResult);
      const exports = symbolsData.symbols as string[];

      if (exports.length > 0) {
        const target = exports[0];
        console.log(`Tracing: ${target}`);

        const traceResult = await handlers.handleTrace({
          sessionId,
          symbol: target,
          args: [],
          maxSteps: 100,
        });

        const traceData = payload(traceResult);
        if (traceData.success === false || traceData.error) {
          const errorMsg = traceData.error || traceData.message || 'unknown error';
          console.log(`Trace failed: ${errorMsg}`);
        } else {
          console.log(`Traced ${traceData.steps?.length ?? 0} steps`);
          if (traceData.steps?.length > 0) {
            console.log('First 3 steps:', traceData.steps.slice(0, 3));
          }
        }
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});
