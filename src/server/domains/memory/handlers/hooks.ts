import type { HardwareBreakpointEngine } from '@native/HardwareBreakpoint';
import type { BreakpointAccess, BreakpointSize } from '@native/HardwareBreakpoint.types';
import type { CodeInjector } from '@native/CodeInjector';
import type { UnifiedProcessManager } from '@server/domains/shared/modules/native';
import type { MCPServerContext } from '@server/MCPServer.context';
import { resolveMemoryDomainPid } from '@server/domains/memory/pid-resolver';
import { handleSafe } from '@server/domains/shared/ResponseBuilder';

export class HookHandlers {
  constructor(
    private readonly bpEngine: HardwareBreakpointEngine | null,
    private readonly injector: CodeInjector,
    private readonly processManager?: UnifiedProcessManager,
    private readonly ctx?: MCPServerContext,
  ) {}

  private async resolvePid(value: unknown): Promise<number> {
    if (!this.processManager) {
      return value as number;
    }
    return await resolveMemoryDomainPid(value, this.processManager, this.ctx);
  }

  async handleBreakpointSet(args: Record<string, unknown>) {
    return handleSafe(async () => {
      if (!this.bpEngine) {
        throw new Error(
          'Hardware breakpoint tools (memory_breakpoint) are only supported on Windows. ' +
            'This tool requires Win32 debug register APIs.',
        );
      }
      const pid = await this.resolvePid(args.pid);
      const config = await this.bpEngine.setBreakpoint(
        pid,
        args.address as string,
        args.access as BreakpointAccess,
        (args.size as BreakpointSize) ?? 4,
      );
      return {
        ...config,
        hint: "Hardware breakpoint set on DR register. Use memory_breakpoint with action='trace' to collect hits.",
      };
    });
  }

  async handleBreakpointRemove(args: Record<string, unknown>) {
    return handleSafe(async () => {
      if (!this.bpEngine) {
        throw new Error(
          'Hardware breakpoint tools (memory_breakpoint) are only supported on Windows. ' +
            'This tool requires Win32 debug register APIs.',
        );
      }
      return {
        removed: await this.bpEngine.removeBreakpoint(args.breakpointId as string),
      };
    });
  }

  async handleBreakpointList(_args: Record<string, unknown>) {
    return handleSafe(async () => {
      if (!this.bpEngine) {
        throw new Error(
          'Hardware breakpoint tools (memory_breakpoint) are only supported on Windows. ' +
            'This tool requires Win32 debug register APIs.',
        );
      }
      const bps = this.bpEngine.listBreakpoints();
      return { breakpoints: bps, count: bps.length };
    });
  }

  async handleBreakpointTrace(args: Record<string, unknown>) {
    return handleSafe(async () => {
      if (!this.bpEngine) {
        throw new Error(
          'Hardware breakpoint tools (memory_breakpoint) are only supported on Windows. ' +
            'This tool requires Win32 debug register APIs.',
        );
      }
      const pid = await this.resolvePid(args.pid);
      const hits = await this.bpEngine.traceAccess(
        pid,
        args.address as string,
        args.access as BreakpointAccess,
        args.maxHits as number | undefined,
        args.timeoutMs as number | undefined,
      );
      return {
        hits,
        hitCount: hits.length,
        hint:
          hits.length > 0
            ? `${hits.length} accesses captured. Check instructionAddress to find the code accessing this address.`
            : 'No hits captured within timeout.',
      };
    });
  }

  async handlePatchBytes(args: Record<string, unknown>) {
    return handleSafe(async () => {
      const pid = await this.resolvePid(args.pid);
      const patch = await this.injector.patchBytes(
        pid,
        args.address as string,
        args.bytes as number[],
      );
      return {
        ...patch,
        hint: `Patch applied. Use memory_patch_undo with patchId "${patch.id}" to restore.`,
      };
    });
  }

  async handlePatchNop(args: Record<string, unknown>) {
    return handleSafe(async () => {
      const pid = await this.resolvePid(args.pid);
      const patch = await this.injector.nopBytes(pid, args.address as string, args.count as number);
      return {
        ...patch,
        hint: `${args.count} bytes NOP'd. Use memory_patch_undo to restore.`,
      };
    });
  }

  async handlePatchUndo(args: Record<string, unknown>) {
    return handleSafe(async () => ({
      restored: await this.injector.unpatch(args.patchId as string),
    }));
  }

  async handleCodeCaves(args: Record<string, unknown>) {
    return handleSafe(async () => {
      const pid = await this.resolvePid(args.pid);
      const caves = await this.injector.findCodeCaves(pid, args.minSize as number | undefined);
      return { caves, count: caves.length };
    });
  }
}
