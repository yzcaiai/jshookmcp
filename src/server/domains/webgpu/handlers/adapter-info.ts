import { handleSafe, type ToolResponse } from '@server/domains/shared/ResponseBuilder';
import { getPageLockManager } from '@modules/webgpu/PageLockManager';
import type { MCPServerContext } from '@server/domains/shared/registry';
import type { WebGPUDomainDependencies, GPUAdapterInfo } from '../types';

/**
 * Handler for webgpu_adapter_info tool
 * Gets GPU adapter information (vendor, architecture, device)
 */
export class AdapterInfoHandler {
  private pageLockManager = getPageLockManager();

  constructor(
    private ctx: MCPServerContext,
    private deps: WebGPUDomainDependencies
  ) {}

  async handle(args: Record<string, unknown>): Promise<ToolResponse> {
    return handleSafe(async () => {
      const page = await this.getActivePage();
      if (!page) {
        throw new Error('No active page. Call browser_launch or browser_attach first.');
      }

      const pageId = page.url();

      // Acquire page lock to prevent concurrent GPU context access
      return await this.pageLockManager.withLock(pageId, async () => {
        const adapterInfo = await page.evaluate(async () => {
          if (!navigator.gpu) {
            throw new Error('WebGPU not available in this browser');
          }

          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) {
            throw new Error('Failed to request GPU adapter');
          }

          const info = adapter.info ?? (adapter as any).requestAdapterInfo?.();

          return {
            vendor: info?.vendor ?? 'unknown',
            architecture: info?.architecture ?? 'unknown',
            device: info?.device ?? 'unknown',
            description: info?.description ?? 'unknown',
          };
        });

        return {
          adapter: adapterInfo,
        };
      });
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
