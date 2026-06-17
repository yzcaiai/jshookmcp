import { handleSafe, type ToolResponse } from '@server/domains/shared/ResponseBuilder';
import { argString } from '@server/domains/shared/parse-args';
import { DetailedDataManager } from '@utils/DetailedDataManager';
import { getShaderDisassemblyCache } from '@modules/webgpu/ShaderCache';
import type { MCPServerContext } from '@server/domains/shared/registry';
import type { WebGPUDomainDependencies } from '../types';

/**
 * Handler for webgpu_shader_disassemble tool
 * Parses WGSL shader into AST and generates human-readable disassembly
 */
export class ShaderDisassembleHandler {
  private ddm: DetailedDataManager;
  private disassemblyCache = getShaderDisassemblyCache();

  constructor(
    private ctx: MCPServerContext,
    private deps: WebGPUDomainDependencies
  ) {
    this.ddm = DetailedDataManager.getInstance();
  }

  async handle(args: Record<string, unknown>): Promise<ToolResponse> {
    return handleSafe(async () => {
      const shaderCode = argString(args, 'shaderCode');
      if (!shaderCode) {
        throw new Error('Missing required argument: shaderCode');
      }

      const format = argString(args, 'format', 'wgsl');
      if (format !== 'wgsl') {
        throw new Error('Only WGSL format is currently supported');
      }

      // Check cache first
      const cached = this.disassemblyCache.get(shaderCode);
      if (cached) {
        return {
          ...cached,
          _cached: true,
        };
      }

      // Report progress for large shaders
      const meta = args['_meta'] as Record<string, unknown> | undefined;
      const progressToken = meta ? argString(meta, 'progressToken') : undefined;

      if (progressToken && shaderCode.length > 10000) {
        this.reportProgress(progressToken, 0.1, 'Parsing shader AST...');
      }

      // Simple AST extraction (real implementation would use @webgpu/wgsl-parser)
      const functions: string[] = [];
      const functionMatches = shaderCode.matchAll(/fn\s+(\w+)/g);
      for (const match of functionMatches) {
        functions.push(match[1]);
      }

      const ast = {
        type: 'Module',
        functions,
      };

      if (progressToken && shaderCode.length > 10000) {
        this.reportProgress(progressToken, 0.5, 'Generating disassembly...');
      }

      const disassembly = this.generateDisassembly(shaderCode);

      if (progressToken && shaderCode.length > 10000) {
        this.reportProgress(progressToken, 1.0, 'Disassembly complete');
      }

      // Check if disassembly is large and should be offloaded
      const result = {
        ast,
        disassembly,
      };

      // Cache the result before offloading
      this.disassemblyCache.set(shaderCode, result);

      return this.ddm.smartHandle(result, 25000);
    });
  }

  private generateDisassembly(shaderCode: string): string {
    // Simple disassembly - real implementation would use proper WGSL parser
    const lines = shaderCode.split('\n');
    return lines
      .map((line, idx) => `${String(idx + 1).padStart(4, ' ')} | ${line}`)
      .join('\n');
  }

  private reportProgress(token: string | undefined, progress: number, message: string): void {
    if (!token || !this.ctx.eventBus) {
      return;
    }

    this.ctx.eventBus.emit('tool:progress', {
      token,
      progress,
      message,
    });
  }
}
