import { logger } from '@utils/logger';

export interface CircuitBreakerState {
  toolName: string;
  failureCount: number;
  lastFailureTime: number;
  state: 'closed' | 'open' | 'half-open';
}

export type CircuitBreakerEventHandler = (event: 'opened' | 'recovered', toolName: string) => void;

const FAILURE_THRESHOLD = 3;
const RECOVERY_MS = 30_000;
const HALF_OPEN_MAX_CALLS = 1;

export class ToolCircuitBreaker {
  private readonly states = new Map<string, CircuitBreakerState>();
  private readonly halfOpenCalls = new Map<string, number>();
  private readonly listeners = new Set<CircuitBreakerEventHandler>();

  onChange(handler: CircuitBreakerEventHandler): void {
    this.listeners.add(handler);
  }

  private emit(event: 'opened' | 'recovered', toolName: string): void {
    for (const handler of this.listeners) {
      try {
        handler(event, toolName);
      } catch {
        // listener errors must not break the breaker
      }
    }
  }

  getState(toolName: string): CircuitBreakerState | undefined {
    return this.states.get(toolName);
  }

  shouldBlock(toolName: string): boolean {
    const entry = this.states.get(toolName);
    if (!entry) return false;

    if (entry.state === 'closed') return false;

    if (entry.state === 'open') {
      const elapsed = Date.now() - entry.lastFailureTime;
      if (elapsed >= RECOVERY_MS) {
        entry.state = 'half-open';
        this.halfOpenCalls.set(toolName, 1);
        logger.info(`[CircuitBreaker] ${toolName}: open → half-open`);
        return false;
      }
      return true;
    }

    if (entry.state === 'half-open') {
      const calls = this.halfOpenCalls.get(toolName) ?? 0;
      if (calls >= HALF_OPEN_MAX_CALLS) return true;
      this.halfOpenCalls.set(toolName, calls + 1);
      return false;
    }

    return false;
  }

  recordSuccess(toolName: string): void {
    const entry = this.states.get(toolName);
    if (!entry) return;

    if (entry.state === 'half-open') {
      entry.state = 'closed';
      entry.failureCount = 0;
      this.halfOpenCalls.delete(toolName);
      logger.info(`[CircuitBreaker] ${toolName}: half-open → closed`);
      this.emit('recovered', toolName);
      return;
    }

    entry.failureCount = 0;
  }

  recordFailure(toolName: string): void {
    let entry = this.states.get(toolName);
    if (!entry) {
      entry = {
        toolName,
        failureCount: 0,
        lastFailureTime: 0,
        state: 'closed',
      };
      this.states.set(toolName, entry);
    }

    entry.failureCount++;
    entry.lastFailureTime = Date.now();

    if (entry.state === 'half-open') {
      entry.state = 'open';
      this.halfOpenCalls.delete(toolName);
      logger.warn(`[CircuitBreaker] ${toolName}: half-open → open (probe failed)`);
      this.emit('opened', toolName);
      return;
    }

    if (entry.state === 'closed' && entry.failureCount >= FAILURE_THRESHOLD) {
      entry.state = 'open';
      logger.warn(
        `[CircuitBreaker] ${toolName}: closed → open (${entry.failureCount} consecutive failures)`,
      );
      this.emit('opened', toolName);
    }
  }

  getStates(): CircuitBreakerState[] {
    return Array.from(this.states.values());
  }

  reset(toolName: string): void {
    this.states.delete(toolName);
    this.halfOpenCalls.delete(toolName);
  }

  getRecoveryMs(): number {
    return RECOVERY_MS;
  }
}
