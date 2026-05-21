import { describe, expect, it, vi } from 'vitest';
import { BrowserTargetSessionManager } from '@modules/browser/BrowserTargetSessionManager';
import { TEST_URLS, withPath } from '@tests/shared/test-urls';

class FakeAttachedSession {
  send = vi.fn(async (method: string) => {
    if (method === 'Runtime.evaluate') {
      return { result: { value: 'attached-result' } };
    }
    if (method === 'Page.addScriptToEvaluateOnNewDocument') {
      return { identifier: 'script-1' };
    }
    return {};
  });

  on() {
    return this;
  }

  off() {
    return this;
  }

  id = vi.fn(() => 'session-1');
  detach = vi.fn(async () => {});
}

class FakeManagedSession {
  constructor(private readonly sessionId: string) {}

  send = vi.fn(async (method: string) => {
    if (method === 'Page.addScriptToEvaluateOnNewDocument') {
      return { identifier: 'script-2' };
    }
    return {};
  });

  on() {
    return this;
  }

  off() {
    return this;
  }

  id = vi.fn(() => this.sessionId);
  detach = vi.fn(async () => {});
}

class FakeParentSession {
  private readonly attachedSession = new FakeAttachedSession();
  readonly childSession = new FakeManagedSession('session-2');
  readonly pageSession = new FakeManagedSession('session-page');
  private readonly listeners = new Map<string, Set<(payload: unknown) => void>>();
  private childSessionMisses = 0;
  private readonly connectionState = {
    session: vi.fn((sessionId: string) => {
      if (sessionId === 'session-1') {
        return this.attachedSession;
      }
      if (sessionId === 'session-page') {
        return this.pageSession;
      }
      if (sessionId === 'session-2') {
        if (this.childSessionMisses > 0) {
          this.childSessionMisses -= 1;
          return null;
        }
        return this.childSession;
      }
      return null;
    }),
  };

  send = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'Target.getTargets') {
      return {
        targetInfos: [
          {
            targetId: 'page-1',
            type: 'page',
            title: 'Main',
            url: TEST_URLS.root,
            attached: false,
          },
          {
            targetId: 'frame-1',
            type: 'iframe',
            title: 'Inner',
            url: withPath(TEST_URLS.root, 'frame'),
            attached: false,
          },
        ],
      };
    }

    if (method === 'Target.attachToTarget') {
      if (params?.targetId === 'page-1') {
        return { sessionId: 'session-page' };
      }
      return { sessionId: 'session-1' };
    }

    return {};
  });

  on(event: string, handler: (payload: unknown) => void) {
    const handlers = this.listeners.get(event) ?? new Set<(payload: unknown) => void>();
    handlers.add(handler);
    this.listeners.set(event, handlers);
    return this;
  }

  off(event: string, handler: (payload: unknown) => void) {
    this.listeners.get(event)?.delete(handler);
    return this;
  }

  detach = vi.fn(async () => {});

  connection = vi.fn(() => this.connectionState);

  setChildSessionLookupMisses(count: number): void {
    this.childSessionMisses = count;
  }

  emit(event: string, payload: unknown): void {
    for (const handler of this.listeners.get(event) ?? []) {
      handler(payload);
    }
  }
}

describe('BrowserTargetSessionManager', () => {
  it('lists targets and supports filtering', async () => {
    const parentSession = new FakeParentSession();
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    const allTargets = await manager.listTargets();
    const iframeTargets = await manager.listTargets({ type: 'iframe' });

    expect(allTargets).toHaveLength(2);
    expect(iframeTargets).toEqual([
      expect.objectContaining({
        targetId: 'frame-1',
        type: 'iframe',
      }),
    ]);
    expect(parentSession.send).toHaveBeenCalledWith('Target.setAutoAttach', {
      autoAttach: true,
      waitForDebuggerOnStart: false,
      flatten: true,
    });
    expect(parentSession.send).toHaveBeenCalledWith('Target.setDiscoverTargets', {
      discover: true,
    });
  });

  it('can skip OOPIF auto-discovery when explicitly disabled', async () => {
    const parentSession = new FakeParentSession();
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    await manager.listTargets({ discoverOOPIF: false });

    expect(parentSession.send).not.toHaveBeenCalledWith('Target.setAutoAttach', expect.anything());
    expect(parentSession.send).not.toHaveBeenCalledWith(
      'Target.setDiscoverTargets',
      expect.anything(),
    );
  });

  it('attaches to a target and evaluates through the flattened session', async () => {
    const parentSession = new FakeParentSession();
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    const target = await manager.attach('frame-1');
    const result = await manager.evaluate('1 + 1');
    await manager.addScriptToEvaluateOnNewDocument('window.__test = 1;');
    const detached = await manager.detach();

    expect(target).toEqual(
      expect.objectContaining({
        targetId: 'frame-1',
        type: 'iframe',
      }),
    );
    expect(result).toBe('attached-result');
    expect(detached).toBe(true);
    expect(parentSession.send).toHaveBeenCalledWith('Target.attachToTarget', {
      targetId: 'frame-1',
      flatten: true,
    });
    expect(parentSession.connection).toHaveBeenCalled();
    expect((parentSession as any).connectionState.session).toHaveBeenCalledWith('session-1');
    expect((parentSession as any).attachedSession.send).toHaveBeenCalledWith('Runtime.evaluate', {
      expression: '1 + 1',
      returnByValue: true,
      awaitPromise: true,
    });
    expect(parentSession.send).toHaveBeenCalledWith('Target.detachFromTarget', {
      sessionId: 'session-1',
    });
    expect((parentSession as any).attachedSession.detach).not.toHaveBeenCalled();
  });

  it('keeps attachment state when flat target detach fails', async () => {
    const parentSession = new FakeParentSession();
    const defaultSend = parentSession.send.getMockImplementation();
    parentSession.send.mockImplementation(async (method: string) => {
      if (method === 'Target.detachFromTarget') {
        throw new Error('detach failed');
      }
      return defaultSend ? await defaultSend(method) : {};
    });
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    await manager.attach('frame-1');

    await expect(manager.detach()).rejects.toThrow('detach failed');
    expect(manager.getAttachedTargetInfo()).toEqual(
      expect.objectContaining({
        targetId: 'frame-1',
      }),
    );
    expect(manager.getAttachedTargetSession()).not.toBeNull();
  });

  it('replays persistent scripts to newly attached managed targets', async () => {
    const parentSession = new FakeParentSession();
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    await manager.listTargets();
    await manager.registerPersistentScript('window.__aiHook = true;', {
      id: 'ai-hook:test',
      evaluateNow: true,
      targetTypes: ['iframe'],
    });

    parentSession.emit('Target.attachedToTarget', {
      sessionId: 'session-2',
      targetInfo: {
        targetId: 'frame-1',
        type: 'iframe',
        title: 'Inner',
        url: withPath(TEST_URLS.root, 'frame'),
        attached: true,
      },
    });

    await vi.waitFor(() => {
      expect(parentSession.childSession.send).toHaveBeenCalledWith(
        'Page.addScriptToEvaluateOnNewDocument',
        expect.objectContaining({
          source: 'window.__aiHook = true;',
        }),
      );
    });
  });

  it('bootstraps existing page targets before registering persistent scripts', async () => {
    const parentSession = new FakeParentSession();
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    const result = await manager.registerPersistentScript('window.__pageHook = true;', {
      id: 'page-hook:test',
      evaluateNow: true,
      targetTypes: ['page'],
    });

    expect(result.appliedTargets).toBeGreaterThanOrEqual(1);
    expect(parentSession.send).toHaveBeenCalledWith('Target.attachToTarget', {
      targetId: 'page-1',
      flatten: true,
    });
    expect(parentSession.pageSession.send).toHaveBeenCalledWith(
      'Page.addScriptToEvaluateOnNewDocument',
      expect.objectContaining({
        source: 'window.__pageHook = true;',
      }),
    );
  });

  it('retries child session lookup before dropping auto-attached targets', async () => {
    const parentSession = new FakeParentSession();
    parentSession.setChildSessionLookupMisses(2);
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    await manager.listTargets();
    await manager.registerPersistentScript('window.__aiHook = true;', {
      id: 'ai-hook:test-retry',
      evaluateNow: true,
      targetTypes: ['iframe'],
    });

    parentSession.emit('Target.attachedToTarget', {
      sessionId: 'session-2',
      targetInfo: {
        targetId: 'frame-1',
        type: 'iframe',
        title: 'Inner',
        url: withPath(TEST_URLS.root, 'frame'),
        attached: true,
      },
    });

    await vi.waitFor(() => {
      expect(parentSession.childSession.send).toHaveBeenCalledWith(
        'Page.addScriptToEvaluateOnNewDocument',
        expect.objectContaining({
          source: 'window.__aiHook = true;',
        }),
      );
    });
  });

  it('does not re-register the same persistent script source on managed targets', async () => {
    const parentSession = new FakeParentSession();
    const browser = {
      target: () => ({
        createCDPSession: vi.fn(async () => parentSession),
      }),
    };
    const manager = new BrowserTargetSessionManager(() => browser as never);

    await manager.registerPersistentScript('window.__aiHook = true;', {
      id: 'ai-hook:dedupe',
      evaluateNow: true,
      targetTypes: ['page'],
    });
    await manager.registerPersistentScript('window.__aiHook = true;', {
      id: 'ai-hook:dedupe',
      evaluateNow: true,
      targetTypes: ['page'],
    });

    const addScriptCalls = parentSession.pageSession.send.mock.calls.filter(
      ([method]) => method === 'Page.addScriptToEvaluateOnNewDocument',
    );
    expect(addScriptCalls).toHaveLength(1);
  });
});
