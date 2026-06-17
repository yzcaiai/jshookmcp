/**
 * CRIT-10: Test Win32-only tools gracefully fail on non-Win32 platforms
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MemoryScanHandlers } from '@server/domains/memory/handlers.impl';

describe('Memory Domain - Win32-only Platform Guards', () => {
  let handlers: MemoryScanHandlers;

  beforeEach(() => {
    // Create handlers with null Win32-only engines (simulating non-Win32 platform)
    const mockScanner = {
      firstScan: vi.fn(),
      nextScan: vi.fn(),
    };
    const mockSessionManager = {
      listSessions: vi.fn(),
      deleteSession: vi.fn(),
    };
    const mockPtrEngine = {
      scan: vi.fn(),
    };
    const mockStructAnalyzer = {
      analyze: vi.fn(),
    };
    const mockInjector = {
      patchBytes: vi.fn(),
      nopBytes: vi.fn(),
      unpatch: vi.fn(),
      findCodeCaves: vi.fn(),
    };
    const mockMemCtrl = {
      writeValue: vi.fn(),
      freeze: vi.fn(),
      dump: vi.fn(),
    };
    const mockEventBus = {
      emit: vi.fn(),
      on: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }),
    };

    handlers = new MemoryScanHandlers(
      mockScanner as any,
      mockSessionManager as any,
      mockPtrEngine as any,
      mockStructAnalyzer as any,
      null, // bpEngine - Win32-only
      mockInjector as any,
      mockMemCtrl as any,
      null, // speedhackEngine - Win32-only
      null, // heapAnalyzer - Win32-only
      null, // peAnalyzer - Win32-only
      null, // antiCheatDetector - Win32-only
      mockEventBus as any,
      undefined,
      undefined,
    );
  });

  describe('Hardware Breakpoint Tools', () => {
    it('memory_breakpoint (set) returns clear error on non-Win32', async () => {
      const result = await handlers.handleBreakpointSet({
        pid: 1234,
        address: '0x400000',
        access: 'write',
      });

      expect(result.content).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            type: 'text',
            text: expect.stringContaining('only supported on Windows'),
          }),
        ]),
      );

      // Verify it's a failure response
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed).toHaveProperty('error');
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_breakpoint (remove) returns clear error on non-Win32', async () => {
      const result = await handlers.handleBreakpointRemove({
        breakpointId: 'bp-123',
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_breakpoint (list) returns clear error on non-Win32', async () => {
      const result = await handlers.handleBreakpointList({});

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_breakpoint (trace) returns clear error on non-Win32', async () => {
      const result = await handlers.handleBreakpointTrace({
        pid: 1234,
        address: '0x400000',
        access: 'read',
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });
  });

  describe('Speedhack Tools', () => {
    it('memory_speedhack (apply) returns clear error on non-Win32', async () => {
      const result = await handlers.handleSpeedhackApply({
        pid: 1234,
        speed: 2.0,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_speedhack (set) returns clear error on non-Win32', async () => {
      const result = await handlers.handleSpeedhackSet({
        pid: 1234,
        speed: 1.5,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });
  });

  describe('Heap Analysis Tools', () => {
    it('memory_heap_enumerate returns clear error on non-Win32', async () => {
      const result = await handlers.handleHeapEnumerate({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_heap_stats returns clear error on non-Win32', async () => {
      const result = await handlers.handleHeapStats({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_heap_anomalies returns clear error on non-Win32', async () => {
      const result = await handlers.handleHeapAnomalies({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });
  });

  describe('PE Analysis Tools', () => {
    it('memory_pe_headers returns clear error on non-Win32', async () => {
      const result = await handlers.handlePEHeaders({
        pid: 1234,
        moduleBase: '0x400000',
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_pe_imports_exports returns clear error on non-Win32', async () => {
      const result = await handlers.handlePEImportsExports({
        pid: 1234,
        moduleBase: '0x400000',
        table: 'both',
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_inline_hook_detect returns clear error on non-Win32', async () => {
      const result = await handlers.handleInlineHookDetect({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });
  });

  describe('Anti-Cheat Detection Tools', () => {
    it('memory_anticheat_detect returns clear error on non-Win32', async () => {
      const result = await handlers.handleAntiCheatDetect({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_guard_pages returns clear error on non-Win32', async () => {
      const result = await handlers.handleGuardPages({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });

    it('memory_integrity_check returns clear error on non-Win32', async () => {
      const result = await handlers.handleIntegrityCheck({
        pid: 1234,
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error).toContain('only supported on Windows');
    });
  });

  describe('Error Message Quality', () => {
    it('error messages mention the specific tool name', async () => {
      const result = await handlers.handleBreakpointSet({
        pid: 1234,
        address: '0x400000',
        access: 'write',
      });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveProperty('success', false);
      expect(parsed.error.toLowerCase()).toMatch(/(hardware breakpoint|memory_breakpoint)/);
    });

    it('error messages do not throw TypeError', async () => {
      // All Win32-only tools should return structured errors, not throw
      const testCases = [
        () => handlers.handleBreakpointSet({ pid: 1234, address: '0x400000', access: 'write' }),
        () => handlers.handleBreakpointRemove({ breakpointId: 'bp-123' }),
        () => handlers.handleBreakpointList({}),
        () => handlers.handleBreakpointTrace({ pid: 1234, address: '0x400000', access: 'read' }),
        () => handlers.handleSpeedhackApply({ pid: 1234, speed: 2.0 }),
        () => handlers.handleSpeedhackSet({ pid: 1234, speed: 1.5 }),
        () => handlers.handleHeapEnumerate({ pid: 1234 }),
        () => handlers.handleHeapStats({ pid: 1234 }),
        () => handlers.handleHeapAnomalies({ pid: 1234 }),
        () => handlers.handlePEHeaders({ pid: 1234, moduleBase: '0x400000' }),
        () => handlers.handlePEImportsExports({ pid: 1234, moduleBase: '0x400000', table: 'both' }),
        () => handlers.handleInlineHookDetect({ pid: 1234 }),
        () => handlers.handleAntiCheatDetect({ pid: 1234 }),
        () => handlers.handleGuardPages({ pid: 1234 }),
        () => handlers.handleIntegrityCheck({ pid: 1234 }),
      ];

      for (const testCase of testCases) {
        const result = await testCase();
        const parsed = JSON.parse(result.content[0].text);
        expect(parsed).toHaveProperty('success', false);
        expect(parsed).toHaveProperty('error');
      }
    });
  });
});
