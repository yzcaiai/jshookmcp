/**
 * Unit tests for ARM64 instruction disassembler (disasm.ts).
 */
import { describe, it, expect } from 'vitest';
import {
  disassembleArm64,
  disassembleInstruction,
  normalizeDisasmArchitecture,
  SUPPORTED_DISASSEMBLY_ARCHITECTURES,
} from '@modules/native-emulator/disasm';

describe('disassembleArm64', () => {
  // ─── RET / NOP ───────────────────────────────────────────────────
  it('decodes RET', () => {
    expect(disassembleArm64(0xd65f03c0, 0x1000n)).toContain('ret');
  });

  it('decodes NOP', () => {
    expect(disassembleArm64(0xd503201f, 0x1000n)).toContain('nop');
  });

  // ─── ADRP ────────────────────────────────────────────────────────
  it('decodes ADRP (from real trace)', () => {
    const result = disassembleArm64(0xb00001e0, 0x1c79cn);
    expect(result).toContain('adrp');
    expect(result).toContain('x0');
  });

  // ─── ADD/SUB immediate ───────────────────────────────────────────
  it('decodes SUB sp, sp, #16', () => {
    const result = disassembleArm64(0xd10043ff, 0x1000n);
    expect(result).toContain('sub');
    expect(result).toContain('#16');
  });

  it('decodes ADD x0, sp, #0', () => {
    const result = disassembleArm64(0x910003e0, 0x1000n);
    expect(result).toContain('add');
    expect(result).toContain('x0');
  });

  it('decodes ADD with shifted immediate (from trace)', () => {
    const result = disassembleArm64(0x91261c00, 0x1c7a0n);
    expect(result).toContain('add');
    expect(result).toContain('x0');
  });

  // ─── MOV (wide immediate) ────────────────────────────────────────
  it('decodes MOVZ as MOV alias', () => {
    // MOVZ w0, #5 → mov w0, #0x5
    // 0 10 100101 00 0000000000000101 00000
    const insn = 0x528000a0;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('mov');
  });

  it('decodes MOVK with shift', () => {
    // MOVK x0, #1, lsl #16
    const insn = 0xf2a00020;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('movk');
    expect(result).toContain('lsl');
  });

  // ─── Bitfield ────────────────────────────────────────────────────
  it('decodes LSR (UBFM alias)', () => {
    // LSR x0, x1, #4 → UBFM x0, x1, #4, #60
    const insn = 0xd340fc20;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('lsr');
  });

  it('decodes SXTB (SBFM alias)', () => {
    // SXTB x0, x1 → SBFM x0, x1, #0, #7
    // Encoding: 1 00 100110 1 000000 000111 00001 00000 = 0x93401C20
    const insn = 0x93401c20;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('sxtb');
  });

  // ─── Branches ────────────────────────────────────────────────────
  it('decodes B (unconditional branch)', () => {
    // B #+4 → imm26=1
    const insn = 0x14000001;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toMatch(/^b\s/);
  });

  it('decodes BL (branch with link)', () => {
    const insn = 0x94000001;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('bl');
  });

  it('decodes B.EQ', () => {
    // B.EQ #+8 → 0x54000100
    const insn = 0x54000100;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('b.eq');
  });

  it('decodes CBZ', () => {
    // CBZ x0, #+8 → 0x34000100
    const insn = 0x34000100;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('cbz');
  });

  it('decodes CBNZ', () => {
    // CBNZ x0, #+8 → 0x35000100
    const insn = 0x35000100;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('cbnz');
  });

  it('decodes BLR', () => {
    const result = disassembleArm64(0xd63f0020, 0x1000n);
    expect(result).toContain('blr');
  });

  it('decodes BR', () => {
    const result = disassembleArm64(0xd61f0060, 0x1000n);
    expect(result).toMatch(/^br\s/);
  });

  // ─── Loads & Stores ──────────────────────────────────────────────
  it('decodes LDR unsigned offset', () => {
    // LDR x0, [sp, #8]  → 0xf94007e0
    const result = disassembleArm64(0xf94007e0, 0x1000n);
    expect(result).toContain('ldr');
    expect(result).toContain('x0');
  });

  it('decodes STR unsigned offset', () => {
    // STR w0, [x1]  → 0xb9000020
    const result = disassembleArm64(0xb9000020, 0x1000n);
    expect(result).toContain('str');
  });

  it('decodes STP (store pair)', () => {
    // STP x29, x30, [sp, #-16]  → 0xa9be7bfd
    const result = disassembleArm64(0xa9be7bfd, 0x1000n);
    expect(result).toContain('stp');
  });

  it('decodes LDP (load pair)', () => {
    // LDP x29, x30, [sp], #16  → 0xa8c17bfd
    const result = disassembleArm64(0xa8c17bfd, 0x1000n);
    expect(result).toContain('ldp');
  });

  // ─── MOV (register) ──────────────────────────────────────────────
  it('decodes MOV (register alias)', () => {
    // ORR x0, xzr, x1 → MOV x0, x1
    // Encoding: sf=1, opc=01, 01010, shift=00, N=0, Rm=1, imm6=0, Rn=31(xzr), Rd=0
    // = 1010 1010 0000 0001 0000 0011 1110 0000 = 0xAA0103E0
    const insn = 0xaa0103e0;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('mov');
    expect(result).toContain('x0, x1');
  });

  // ─── CSEL ────────────────────────────────────────────────────────
  it('decodes CSEL', () => {
    // CSEL x0, x1, x2, eq → 0x9A820020
    const insn = 0x9a820020;
    const result = disassembleArm64(insn, 0x1000n);
    expect(result).toContain('csel');
    expect(result).toContain('eq');
  });

  // ─── MRS ─────────────────────────────────────────────────────────
  it('decodes MRS NZCV', () => {
    // MRS x0, NZCV → 0xd53b4200
    const result = disassembleArm64(0xd53b4200, 0x1000n);
    expect(result).toContain('mrs');
  });

  // ─── Unknown instructions ────────────────────────────────────────
  it('returns unknown for unrecognized opcodes', () => {
    const result = disassembleArm64(0xdeadbeef, 0x1000n);
    expect(result).toContain('<unknown>');
    expect(result).toContain('deadbeef');
  });

  // ─── Regression: real trace from 91porn APK ──────────────────────
  it('decodes all instructions from real trace', () => {
    const trace = [
      { op: 0xb00001e0, pc: 0x1c798n, expect: 'adrp' },
      { op: 0x91261c00, pc: 0x1c7a0n, expect: 'add' },
      { op: 0xd65f03c0, pc: 0x1c7a4n, expect: 'ret' },
    ];

    for (const { op, pc, expect: mnemonic } of trace) {
      const result = disassembleArm64(op, pc);
      expect(result).toContain(mnemonic);
    }
  });
});

describe('disassembleInstruction multi-architecture entrypoint', () => {
  it('exposes stable architecture aliases through the public facade', () => {
    expect(SUPPORTED_DISASSEMBLY_ARCHITECTURES).toEqual([
      'arm64',
      'aarch64',
      'x86',
      'x64',
      'riscv32',
      'riscv64',
      'mips',
      'mips32',
      'mipsel',
    ]);
    expect(normalizeDisasmArchitecture('aarch64')).toBe('arm64');
    expect(normalizeDisasmArchitecture('riscv64')).toBe('riscv');
    expect(normalizeDisasmArchitecture('mips32')).toBe('mips');
  });

  it('rejects too-short fixed-width byte input at the registry boundary', () => {
    expect(() => disassembleInstruction('riscv64', [0x13, 0x00, 0x00], 0x1000n)).toThrow(
      /at least 4 bytes/i,
    );
  });

  it('keeps ARM64 compatibility through the generic entrypoint', () => {
    const result = disassembleInstruction('arm64', 0xd65f03c0, 0x1000n);
    expect(result).toContain('ret');
  });

  it('decodes x64 common instructions', () => {
    expect(disassembleInstruction('x64', [0x90], 0x1000n)).toContain('nop');
    expect(disassembleInstruction('x64', [0x55], 0x1000n)).toContain('push');
    expect(disassembleInstruction('x64', [0x55], 0x1000n)).toContain('rbp');
    expect(disassembleInstruction('x64', [0x48, 0x89, 0xe5], 0x1000n)).toContain('mov');
    expect(disassembleInstruction('x64', [0x48, 0x89, 0xe5], 0x1000n)).toContain('rbp, rsp');
  });

  it('decodes x86 relative and register instructions', () => {
    expect(disassembleInstruction('x86', [0xe8, 0x01, 0x00, 0x00, 0x00], 0x1000n)).toContain(
      '0x1006',
    );
    expect(disassembleInstruction('x86', [0x31, 0xc0], 0x1000n)).toContain('xor');
    expect(disassembleInstruction('x86', [0x31, 0xc0], 0x1000n)).toContain('eax, eax');
  });

  it('decodes modern x86/x64 SIMD and crypto prefixes', () => {
    expect(disassembleInstruction('x64', [0x0f, 0x58, 0xc1], 0x1000n)).toContain('addps');
    expect(disassembleInstruction('x64', [0x66, 0x0f, 0xef, 0xc0], 0x1000n)).toContain('pxor');
    expect(disassembleInstruction('x64', [0x66, 0x0f, 0x38, 0xdc, 0xc1], 0x1000n)).toContain(
      'aesenc',
    );
    expect(disassembleInstruction('x64', [0x66, 0x0f, 0x3a, 0x44, 0xc1, 0x00], 0x1000n)).toContain(
      'pclmulqdq',
    );
  });

  it('decodes AVX and AVX2 VEX-prefixed instructions', () => {
    expect(disassembleInstruction('x64', [0xc5, 0xf4, 0x58, 0xc2], 0x1000n)).toContain('vaddps');
    expect(disassembleInstruction('x64', [0xc4, 0xe2, 0x7d, 0x40, 0xc1], 0x1000n)).toContain(
      'vpmulld',
    );
  });

  it('decodes common AVX-512 EVEX-prefixed instructions', () => {
    expect(disassembleInstruction('x64', [0x62, 0xf1, 0x74, 0x48, 0x58, 0xc2], 0x1000n)).toContain(
      'vaddps',
    );
    expect(disassembleInstruction('x64', [0x62, 0xf2, 0x75, 0x48, 0x25, 0xc2], 0x1000n)).toContain(
      'vpternlogd',
    );
    expect(disassembleInstruction('x64', [0x62, 0xf2, 0x7d, 0x48, 0x7c, 0xc0], 0x1000n)).toContain(
      'vpbroadcastd',
    );
  });

  it('decodes RISC-V common instructions', () => {
    expect(disassembleInstruction('riscv64', 0x00000013, 0x1000n)).toContain('nop');
    expect(disassembleInstruction('riscv64', 0x00008067, 0x1000n)).toContain('ret');
    expect(disassembleInstruction('riscv64', 0x00100093, 0x1000n)).toContain('addi');
    expect(disassembleInstruction('riscv64', 0x002081b3, 0x1000n)).toContain('add');
  });

  it('decodes MIPS common instructions', () => {
    expect(disassembleInstruction('mips', 0x00000000, 0x1000n)).toContain('nop');
    expect(disassembleInstruction('mips', 0x012a4020, 0x1000n)).toContain('add');
    expect(disassembleInstruction('mips', 0x8d080004, 0x1000n)).toContain('lw');
  });

  it('decodes MIPSEL bytes using little-endian order', () => {
    const result = disassembleInstruction('mipsel', [0x04, 0x00, 0x08, 0x8d], 0x1000n);
    expect(result).toContain('lw');
    expect(result).toContain('$t0');
  });
});
