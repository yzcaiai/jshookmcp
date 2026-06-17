import { handleSafe, type ToolResponse } from '@server/domains/shared/ResponseBuilder';
import { argNumber, argBool, argString } from '@server/domains/shared/parse-args';
import { getPageLockManager } from '@modules/webgpu/PageLockManager';
import type { MCPServerContext } from '@server/domains/shared/registry';
import type { WebGPUDomainDependencies, TimingStats } from '../types';

/**
 * Handler for webgpu_timing_analysis tool
 * GPU timing analysis for side-channel detection (measures variance)
 */
export class TimingAnalysisHandler {
  private pageLockManager = getPageLockManager();

  constructor(
    private ctx: MCPServerContext,
    private deps: WebGPUDomainDependencies
  ) {}

  async handle(args: Record<string, unknown>): Promise<ToolResponse> {
    return handleSafe(async () => {
      const iterations = argNumber(args, 'iterations');
      if (!iterations || iterations <= 0) {
        throw new Error('Missing or invalid required argument: iterations (must be > 0)');
      }

      const detectAnomalies = argBool(args, 'detectAnomalies', false);
      const meta = args['_meta'] as Record<string, unknown> | undefined;
      const progressToken = meta ? argString(meta, 'progressToken') : undefined;

      const page = await this.getActivePage();
      if (!page) {
        throw new Error('No active page. Call browser_launch or browser_attach first.');
      }

      const pageId = page.url();

      // Acquire page lock to prevent concurrent GPU context access
      return await this.pageLockManager.withLock(pageId, async () => {
        const stats = await page.evaluate(
          async ({ iterations, detectAnomalies }: { iterations: number; detectAnomalies: boolean }) => {
            if (!navigator.gpu) {
              throw new Error('WebGPU not available');
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
              throw new Error('Failed to request GPU adapter');
            }

            const device = await adapter.requestDevice();
            const timings: number[] = [];

            for (let i = 0; i < iterations; i++) {
              const start = performance.now();

              // Simple GPU timing test: create buffer and wait for completion
              const buffer = device.createBuffer({
                size: 1024,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              });

              await device.queue.onSubmittedWorkDone();

              const end = performance.now();
              timings.push(end - start);

              buffer.destroy();

              // Report progress every 20%
              if ((window as any).__webgpuProgressCallback && i % Math.ceil(iterations / 5) === 0) {
                (window as any).__webgpuProgressCallback(i / iterations);
              }
            }

            const mean = timings.reduce((a, b) => a + b, 0) / timings.length;
            const variance =
              timings.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / timings.length;
            const stddev = Math.sqrt(variance);
            const min = Math.min(...timings);
            const max = Math.max(...timings);

            const result: any = {
              timings,
              mean,
              stddev,
              min,
              max,
            };

            if (detectAnomalies) {
              const threshold = 2.0; // 2 standard deviations
              result.anomalies = timings
                .map((val, idx) => ({
                  index: idx,
                  value: val,
                  deviation: Math.abs(val - mean) / stddev,
                }))
                .filter((a) => a.deviation > threshold);
            }

            return result;
          },
          { iterations, detectAnomalies }
        );

        return stats;
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
