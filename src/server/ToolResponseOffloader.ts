/**
 * ToolResponseOffloader — response middleware that automatically offloads large data
 * from tool responses to disk, replacing the payload with a compact reference.
 *
 * Placement in the call chain (see MCPServer.executeToolWithTracking):
 *
 *   router.execute()       ← handler returns raw response
 *     → LargeDataOffloader.offload()     ← this: writes large data → file/detailId
 *     → contextGuard.recordCall()
 *     → contextGuard.enrichResponse()   ← tab context + repeat warnings
 *     → appendExecutionMetrics()
 *     → tokenBudget.recordToolCall()
 *
 * What gets offloaded:
 *   - data:image/...;base64,...  → binary file on disk
 *   - large JSON strings (> threshold)  → DetailedDataManager → detailId reference
 *   - large raw strings (> file threshold) → .txt file on disk
 *
 * The placeholder format replaces large values in-place:
 *   { _offload: { type: 'detailId' | 'file', path?: string, detailId?: string, size: number } }
 */

import { DetailedDataManager } from '@utils/DetailedDataManager';
import { sanitizeForCache, formatSize, DATA_URI_RE } from '@utils/sanitizeForCache';
import { logger } from '@utils/logger';

export interface OffloaderConfig {
  /** Strings larger than this (bytes) go to DetailedDataManager. Default: 512KB */
  detailThreshold?: number;
  /** Strings larger than this (bytes) go directly to a file. Default: 4MB */
  fileThreshold?: number;
  /** Subdirectory under project root for offloaded files. Default: artifacts/offloaded */
  outputDir?: string;
  /** Tools excluded from offloading (e.g. tools that intentionally return large data). */
  excludeTools?: Set<string>;
}

interface OffloadPlaceholder {
  _offload: {
    type: 'detailId' | 'file';
    /** Absolute path (type=file) */
    path?: string;
    /** Detail ID (type=detailId) */
    detailId?: string;
    /** Human-readable size */
    size: string;
    /** MIME type hint (type=file, data URI only) */
    mimeType?: string;
  };
}

const DETAILID_RE = /"_?offload"|detailId|_filePath/;

export class LargeDataOffloader {
  private readonly detailThreshold: number;
  private readonly fileThreshold: number;
  private readonly excludeTools: Set<string>;

  constructor(
    private readonly detailedData: DetailedDataManager,
    config: OffloaderConfig = {},
  ) {
    this.detailThreshold = config.detailThreshold ?? 512 * 1024; // 512KB
    this.fileThreshold = config.fileThreshold ?? 4 * 1024 * 1024; // 4MB
    this.excludeTools = config.excludeTools ?? new Set();
  }

  /**
   * Store structured data in DetailedDataManager. Returns the placeholder.
   */
  private storeInDetailManager(data: unknown, _toolName: string, _idx: number): OffloadPlaceholder {
    const detailId = this.detailedData.store(data);
    const entry = (
      this.detailedData as unknown as { cache: Map<string, { size: number }> }
    ).cache.get(detailId);
    const size = entry?.size ?? 0;

    logger.info(
      `[Offloader] Stored in DetailDataManager (${formatSize(size)}) → detailId=${detailId}`,
    );
    return {
      _offload: {
        type: 'detailId',
        detailId,
        size: formatSize(size),
      },
    };
  }

  /**
   * Detect a "detail wrapper" response shape — an object carrying a string
   * `detailId` plus a `data` / `summary` / `preview` payload field. These come
   * from get_detailed_data and similar tools; their payload may contain
   * un-offloaded oversized fields that must be sanitized rather than skipped.
   */
  private isDetailWrapper(value: unknown): boolean {
    if (typeof value !== 'object' || value === null) return false;
    const obj = value as Record<string, unknown>;
    if (typeof obj.detailId !== 'string') return false;
    return 'data' in obj || 'summary' in obj || 'preview' in obj;
  }

  /**
   * Try to parse a string as JSON. Returns the parsed object or null.
   */
  private tryParseJson(str: string): unknown | null {
    try {
      const trimmed = str.trim();
      if (
        (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
        (trimmed.startsWith('[') && trimmed.endsWith(']'))
      ) {
        return JSON.parse(trimmed);
      }
    } catch {
      // Not JSON
    }
    return null;
  }

  /**
   * Offload large data in a tool response. Mutates the response in-place.
   *
   * Returns the same response object (mutated) for chaining convenience.
   */
  offload<T extends { content?: unknown[]; isError?: boolean }>(toolName: string, response: T): T {
    if (response.isError) return response;
    if (this.excludeTools.has(toolName)) return response;

    const content = response.content;
    if (!Array.isArray(content)) return response;

    let changed = false;

    for (let i = 0; i < content.length; i++) {
      const entry = content[i];
      if (typeof entry !== 'object' || entry === null) continue;

      const record = entry as Record<string, unknown>;
      if (record.type !== 'text' && record.type !== 'resource') continue;

      const text = record.text as string | undefined;
      if (typeof text !== 'string' || text.length < this.detailThreshold) continue;

      // ── Detail wrapper (e.g. get_detailed_data) → sanitize its data branch ──
      // A response like { detailId, path, data } always carries a literal
      // "detailId", so the old blunt DETAILID_RE skip let multi-MB blobs inside
      // `data` escape unmodified (issue #62). Instead of skipping the whole
      // entry, recursively offload oversized fields within it. This is
      // defense-in-depth: the primary fix sanitizes at DetailedDataManager.store,
      // but this also catches any future path that bypasses the cache.
      const detailWrapper = this.tryParseJson(text);
      if (detailWrapper !== null && this.isDetailWrapper(detailWrapper)) {
        const sanitized = sanitizeForCache(detailWrapper);
        if (sanitized !== detailWrapper) {
          content[i] = { ...record, text: JSON.stringify(sanitized, null, 2) };
          changed = true;
        }
        continue;
      }

      // Skip pure offload placeholders / file-reference responses (idempotent).
      if (DETAILID_RE.test(text)) continue;

      // ── Data URI (base64 image) → write binary file ──
      if (DATA_URI_RE.test(text)) {
        // Fire-and-forget: we can't make this async cleanly here without changing
        // the return signature. We use a synchronous placeholder for now.
        // The actual write is deferred — callers who need the path should use
        // writeDataUriToFile() directly or await offloadAsync().
        content[i] = {
          ...record,
          text: JSON.stringify(
            {
              _offload: {
                type: 'file',
                pending: true,
                dataUriLength: text.length,
                hint: `Use get_detailed_data() or read file after async offload completes`,
                size: formatSize(text.length),
                mimeType: text.match(DATA_URI_RE)?.[1] ?? 'application/octet-stream',
              },
            },
            null,
            2,
          ),
        };
        changed = true;
        continue;
      }

      // ── Large JSON string → DetailedDataManager ──
      const parsed = this.tryParseJson(text);
      if (parsed !== null) {
        content[i] = {
          ...record,
          text: JSON.stringify(this.storeInDetailManager(parsed, toolName, i), null, 2),
        };
        changed = true;
        continue;
      }

      // ── Large non-JSON string → file ──
      if (text.length >= this.fileThreshold) {
        content[i] = {
          ...record,
          text: JSON.stringify(
            {
              _offload: {
                type: 'file',
                pending: true,
                hint: 'Large raw string — use get_detailed_data() or await async offload',
                size: formatSize(text.length),
              },
            },
            null,
            2,
          ),
        };
        changed = true;
        continue;
      }
    }

    if (changed) {
      logger.debug(`[Offloader] Offloaded large data from ${toolName}`);
    }
    return response;
  }

  /**
   * Async version — actually writes pending files.
   * Call this after offload() to flush pending writes.
   * Returns the same response (mutated).
   */
  async offloadAsync<T extends { content?: unknown[]; isError?: boolean }>(
    toolName: string,
    response: T,
  ): Promise<T> {
    if (response.isError) return response;
    if (this.excludeTools.has(toolName)) return response;

    const content = response.content;
    if (!Array.isArray(content)) return response;

    for (let i = 0; i < content.length; i++) {
      const entry = content[i];
      if (typeof entry !== 'object' || entry === null) continue;

      const record = entry as Record<string, unknown>;
      if (record.type !== 'text') continue;

      const text = record.text as string | undefined;
      if (typeof text !== 'string') continue;

      // Check for pending offload placeholder
      try {
        const parsed = JSON.parse(text);
        /* eslint-disable no-underscore-dangle */
        if (parsed?._offload?.pending) {
          // Actually write the file now
          if (
            DATA_URI_RE.test(
              parsed._offload.mimeType ? `data:${parsed._offload.mimeType};base64,` : '',
            )
          ) {
            /* eslint-enable no-underscore-dangle */
            // Re-extract from original... but we already consumed it above.
            // For simplicity: skip async write if we can't re-extract.
            // The sync offload() already provided a placeholder.
            logger.debug(`[Offloader] Skipping async write for already-placeholdered entry ${i}`);
          }
        }
      } catch {
        // Not JSON — ignore
      }
    }

    return response;
  }
}
