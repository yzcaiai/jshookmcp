#!/usr/bin/env node
/**
 * Quick diagnosis: check if libmmkv.so has a relocated init_array slot that points to -8.
 */
import { writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { extractArm64Libs } from '../src/modules/native-emulator/apk.ts';
import { NativeEmulator } from '../src/modules/native-emulator/NativeEmulator.ts';

console.log('=== libmmkv.so .init_array diagnostic ===\n');

const libs = await extractArm64Libs('D:/cumhub/reverse/luolishe/luoys-6.10.apk');
const mmkv = libs.find((l) => l.name === 'libmmkv.so');

if (!mmkv) {
  console.error('libmmkv.so not found');
  process.exit(1);
}

const tmpDir = await mkdtemp(join(tmpdir(), 'mmkv-diag-'));
const soPath = join(tmpDir, 'libmmkv.so');
await writeFile(soPath, mmkv.bytes);

try {
  const emu = new NativeEmulator({ syscalls: false });

  console.log('Loading libmmkv.so...');
  const result = emu.loadLibrary(mmkv.bytes);

  console.log('✅ Load succeeded!');
  console.log(`Entry: 0x${result.entry.toString(16)}`);
  console.log(`Unresolved imports: ${result.unresolvedImports.length}`);
  console.log(`Constructor faults: ${result.constructorFaults.length}`);

  if (result.constructorFaults.length > 0) {
    console.log('\nConstructor faults:');
    result.constructorFaults.forEach((f, i) => console.log(`  ${i + 1}. ${f}`));
  }
} catch (err) {
  console.log(`\n❌ Load failed: ${err.message}`);
  console.log('\nThis is the bug we need to fix!');
  console.log('Stack trace:');
  console.log(err.stack);
} finally {
  await rm(tmpDir, { recursive: true, force: true });
}
