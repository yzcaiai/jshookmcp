import { afterAll, describe, expect, it } from 'vitest';
import { rmSync } from 'node:fs';
import { LargeDataOffloader } from '@server/ToolResponseOffloader';
import { DetailedDataManager } from '@utils/DetailedDataManager';
import { getOffloadDir } from '@utils/sanitizeForCache';

function textResponse(obj: unknown) {
  return {
    content: [{ type: 'text' as const, text: JSON.stringify(obj, null, 2) }],
  };
}

describe('LargeDataOffloader — issue #62 structural detection', () => {
  const offloadDir = getOffloadDir();

  afterAll(() => {
    // Best-effort cleanup of any files written by these tests.
    rmSync(offloadDir, { recursive: true, force: true });
  });

  it('sanitizes a data: URI nested in a get_detailed_data wrapper instead of skipping it', () => {
    const offloader = new LargeDataOffloader(new DetailedDataManager());
    const dataUri = 'data:image/png;base64,' + 'A'.repeat(2 * 1024 * 1024);
    const response = textResponse({
      success: true,
      detailId: 'detail_abc',
      path: 'requests',
      data: [{ url: dataUri, method: 'GET', requestId: 'r1' }],
    });

    offloader.offload('get_detailed_data', response);

    const text = (response.content[0] as { text: string }).text;
    // The wrapper is NOT skipped: the multi-MB blob is gone, replaced by a placeholder.
    expect(text).toContain('_offload');
    expect(text).not.toContain('A'.repeat(500));
    expect(text.length).toBeLessThan(5000);
    // Wrapper metadata survives.
    expect(text).toContain('detail_abc');
  });

  it('still skips a pure offload placeholder response (idempotent)', () => {
    const offloader = new LargeDataOffloader(new DetailedDataManager());
    // A response that is already an offload placeholder padded over the threshold.
    const placeholder = {
      _offload: { type: 'file', path: 'artifacts/offloaded/x.bin', size: '2.0MB' },
      padding: 'p'.repeat(600 * 1024),
    };
    const response = textResponse(placeholder);
    const before = (response.content[0] as { text: string }).text;

    offloader.offload('some_tool', response);

    const after = (response.content[0] as { text: string }).text;
    // Unchanged — the pure placeholder branch is left alone.
    expect(after).toBe(before);
  });

  it('leaves small responses untouched', () => {
    const offloader = new LargeDataOffloader(new DetailedDataManager());
    const response = textResponse({ success: true, detailId: 'd1', data: { ok: true } });
    const before = (response.content[0] as { text: string }).text;

    offloader.offload('get_detailed_data', response);

    expect((response.content[0] as { text: string }).text).toBe(before);
  });
});
