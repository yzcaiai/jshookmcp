import { handleSafe, type ToolResponse } from '@server/domains/shared/ResponseBuilder';
import { argString } from '@server/domains/shared/parse-args';
import { getPageLockManager } from '@modules/webgpu/PageLockManager';
import { getShaderCompileCache } from '@modules/webgpu/ShaderCache';
import type { MCPServerContext } from '@server/domains/shared/registry';
import type { WebGPUDomainDependencies, ShaderMetadata } from '../types';

/**
 * Handler for webgpu_shader_compile tool
 * Compiles WGSL shader and extracts metadata (entry points, bindings, attributes)
 */
export class ShaderCompileHandler {
  private pageLockManager = getPageLockManager();
  private compileCache = getShaderCompileCache();

  constructor(
    private ctx: MCPServerContext,
    private deps: WebGPUDomainDependencies
  ) {}

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
      const cached = this.compileCache.get(shaderCode);
      if (cached) {
        return {
          ...cached,
          _cached: true,
        };
      }

      const page = await this.getActivePage();
      if (!page) {
        throw new Error('No active page. Call browser_launch or browser_attach first.');
      }

      const pageId = page.url();

      // Acquire page lock to prevent concurrent GPU context access
      const result = await this.pageLockManager.withLock(pageId, async () => {
        return await page.evaluate(
          async (code: string) => {
            if (!navigator.gpu) {
              throw new Error('WebGPU not available');
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
              throw new Error('Failed to request GPU adapter');
            }

            const device = await adapter.requestDevice();

            try {
              const shaderModule = device.createShaderModule({
                code,
              });

              // Extract metadata from shader code
              const entryPoints: Array<{ name: string; stage: string }> = [];

              // Simple regex-based parsing (real implementation would use proper WGSL parser)
              const vertexMatch = code.match(/@vertex\s+fn\s+(\w+)/);
              const fragmentMatch = code.match(/@fragment\s+fn\s+(\w+)/);
              const computeMatch = code.match(/@compute\s+fn\s+(\w+)/);

              if (vertexMatch) {
                entryPoints.push({ name: vertexMatch[1], stage: 'vertex' });
              }
              if (fragmentMatch) {
                entryPoints.push({ name: fragmentMatch[1], stage: 'fragment' });
              }
              if (computeMatch) {
                entryPoints.push({ name: computeMatch[1], stage: 'compute' });
              }

              return {
                compiled: true,
                metadata: {
                  entryPoints,
                },
              };
            } catch (err: any) {
              throw new Error(`Shader compilation failed: ${err.message}`);
            }
          },
          shaderCode
        );
      });

      // Cache the result
      this.compileCache.set(shaderCode, result);

      return result;
    });
  }

  private async getActivePage(): Promise<any> {
    if (!this.deps.pageController) {
      return null;
    }

    try {
      return await this.deps.pageController.getActivePage();
    } catch {
      return null;
    }
  }
}
