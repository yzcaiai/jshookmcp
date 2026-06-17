/**
 * InjectionValidator — Target process and payload validation for DLL/shellcode injection
 *
 * Provides configurable validation layers:
 * - Target process validation (PID, accessibility, signature)
 * - Payload validation (file existence, format, size, optional hash)
 * - Confirmation requirements (based on validation mode)
 */

import { existsSync, statSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { logger } from '@utils/logger';

export enum InjectionValidationMode {
  STRICT = 'strict',       // All checks enforced, confirmation required for unsigned processes
  BALANCED = 'balanced',   // Validation + warnings, no confirmation required (default)
  PERMISSIVE = 'permissive', // Basic checks only (PID exists, file exists)
  DISABLED = 'disabled',   // No validation (backward compatible)
}

export interface InjectionValidatorConfig {
  mode: InjectionValidationMode;
  maxShellcodeSize: number;  // bytes
  requireHashForDll: boolean;
}

export interface TargetProcessValidation {
  valid: boolean;
  processExists: boolean;
  processAccessible: boolean;
  isCriticalSystemProcess: boolean;
  processName?: string;
  executablePath?: string;
  isSigned?: boolean;         // Win32 only
  signatureValid?: boolean;   // Win32 only
  warnings: string[];
  errors: string[];
}

export interface PayloadValidation {
  valid: boolean;
  payloadType: 'dll' | 'shellcode';

  // DLL-specific
  fileExists?: boolean;
  fileSize?: number;
  expectedHash?: string;
  actualHash?: string;
  hashMatch?: boolean;

  // Shellcode-specific
  shellcodeSize?: number;
  encodingValid?: boolean;

  warnings: string[];
  errors: string[];
}

export interface ConfirmationRequirement {
  required: boolean;
  reason?: string;
  overrideFlag?: string;
}

// Critical system processes that should never be injected into
const CRITICAL_SYSTEM_PROCESSES: Record<string, string[]> = {
  win32: ['csrss.exe', 'smss.exe', 'winlogon.exe', 'services.exe', 'lsass.exe', 'wininit.exe', 'System'],
  linux: ['init', 'systemd', 'kthreadd'],
  darwin: ['launchd', 'kernel_task'],
};

const MAX_REASONABLE_DLL_SIZE = 50 * 1024 * 1024; // 50MB
const MIN_REASONABLE_SHELLCODE_SIZE = 4; // bytes

export class InjectionValidator {
  private config: InjectionValidatorConfig;

  constructor(config: InjectionValidatorConfig) {
    this.config = config;
  }

  /**
   * Validate target process for injection
   */
  async validateTargetProcess(pid: number): Promise<TargetProcessValidation> {
    const result: TargetProcessValidation = {
      valid: true,
      processExists: false,
      processAccessible: false,
      isCriticalSystemProcess: false,
      warnings: [],
      errors: [],
    };

    // DISABLED mode: skip all checks
    if (this.config.mode === InjectionValidationMode.DISABLED) {
      result.processExists = true;
      result.processAccessible = true;
      return result;
    }

    // Basic PID validation
    if (!Number.isInteger(pid) || pid <= 0) {
      result.valid = false;
      result.errors.push(`Invalid PID: ${pid}`);
      return result;
    }

    // Check if process exists and is accessible
    try {
      const { UnifiedProcessManager } = await import('@modules/process/ProcessManager.impl');
      const processMgr = new UnifiedProcessManager();
      const processInfo = await processMgr.getProcessByPid(pid);

      if (!processInfo) {
        result.valid = false;
        result.errors.push(`Process ${pid} does not exist or is not accessible`);
        return result;
      }

      result.processExists = true;
      result.processAccessible = true;
      result.processName = processInfo.name;
      result.executablePath = processInfo.path;

      // Check if it's a critical system process
      const platform = processMgr.getPlatform();
      const criticalProcesses = CRITICAL_SYSTEM_PROCESSES[platform] ?? [];
      const processNameLower = processInfo.name?.toLowerCase() ?? '';

      if (criticalProcesses.some(critical => processNameLower.includes(critical.toLowerCase()))) {
        result.valid = false;
        result.isCriticalSystemProcess = true;
        result.errors.push(`Target process is a critical system process: ${processInfo.name}`);
        return result;
      }

      // Platform-specific signature validation (Win32 only in STRICT/BALANCED mode)
      if (platform === 'win32' && this.config.mode !== InjectionValidationMode.PERMISSIVE) {
        const signatureCheck = await this.checkWindowsSignature(processInfo.path);
        result.isSigned = signatureCheck.isSigned;
        result.signatureValid = signatureCheck.isValid;

        if (!signatureCheck.isSigned && this.config.mode === InjectionValidationMode.BALANCED) {
          result.warnings.push(`Process is not digitally signed: ${processInfo.name}`);
        }
      }

    } catch (error) {
      result.valid = false;
      result.errors.push(`Failed to validate process: ${error instanceof Error ? error.message : String(error)}`);
      logger.debug('Process validation error:', error);
    }

    return result;
  }

  /**
   * Validate DLL payload
   */
  async validateDllPayload(
    dllPath: string,
    options?: { expectedHash?: string }
  ): Promise<PayloadValidation> {
    const result: PayloadValidation = {
      valid: true,
      payloadType: 'dll',
      warnings: [],
      errors: [],
    };

    // DISABLED mode: skip all checks
    if (this.config.mode === InjectionValidationMode.DISABLED) {
      return result;
    }

    // Check file existence
    try {
      result.fileExists = existsSync(dllPath);

      if (!result.fileExists) {
        result.valid = false;
        result.errors.push(`DLL file not found: ${dllPath}`);
        return result;
      }

      // Get file size
      const stats = statSync(dllPath);
      result.fileSize = stats.size;

      // Warn on unusually large files
      if (result.fileSize > MAX_REASONABLE_DLL_SIZE) {
        result.warnings.push(`DLL file is unusually large (${Math.round(result.fileSize / 1024 / 1024)}MB). Verify this is correct.`);
      }

      // Hash validation
      if (this.config.requireHashForDll && !options?.expectedHash) {
        result.valid = false;
        result.errors.push('DLL hash validation required but no expectedHash provided');
      }

      if (options?.expectedHash) {
        result.expectedHash = options.expectedHash;
        result.actualHash = this.computeFileHash(dllPath);
        result.hashMatch = result.actualHash === options.expectedHash;

        if (!result.hashMatch) {
          result.valid = false;
          result.errors.push('DLL hash mismatch');
        }
      }

    } catch (error) {
      result.valid = false;
      result.errors.push(`Failed to validate DLL: ${error instanceof Error ? error.message : String(error)}`);
      logger.debug('DLL validation error:', error);
    }

    return result;
  }

  /**
   * Validate shellcode payload
   */
  async validateShellcodePayload(
    shellcode: string,
    encoding: 'hex' | 'base64'
  ): Promise<PayloadValidation> {
    const result: PayloadValidation = {
      valid: true,
      payloadType: 'shellcode',
      encodingValid: false,
      warnings: [],
      errors: [],
    };

    // DISABLED mode: skip all checks
    if (this.config.mode === InjectionValidationMode.DISABLED) {
      result.encodingValid = true;
      return result;
    }

    // Check for empty shellcode
    if (!shellcode || shellcode.length === 0) {
      result.valid = false;
      result.errors.push('Shellcode is empty');
      return result;
    }

    try {
      let shellcodeBytes: Buffer;

      // Validate encoding and decode
      if (encoding === 'hex') {
        const cleanHex = shellcode.replace(/\s/g, '');
        if (!/^[0-9A-Fa-f]*$/.test(cleanHex)) {
          result.valid = false;
          result.errors.push('Invalid hex encoding in shellcode');
          return result;
        }
        shellcodeBytes = Buffer.from(cleanHex, 'hex');
      } else {
        try {
          shellcodeBytes = Buffer.from(shellcode, 'base64');
          // Verify it's valid base64 by re-encoding and comparing
          const reEncoded = shellcodeBytes.toString('base64');
          if (reEncoded !== shellcode.replace(/\s/g, '')) {
            result.valid = false;
            result.errors.push('Invalid base64 encoding in shellcode');
            return result;
          }
        } catch {
          result.valid = false;
          result.errors.push('Invalid base64 encoding in shellcode');
          return result;
        }
      }

      result.encodingValid = true;
      result.shellcodeSize = shellcodeBytes.length;

      // Size validation
      if (result.shellcodeSize > this.config.maxShellcodeSize) {
        result.valid = false;
        result.errors.push(
          `Shellcode size (${result.shellcodeSize} bytes) exceeds maximum allowed (${this.config.maxShellcodeSize} bytes)`
        );
      }

      // Warn on suspiciously small shellcode
      if (result.shellcodeSize < MIN_REASONABLE_SHELLCODE_SIZE) {
        result.warnings.push(
          `Shellcode is suspiciously small (${result.shellcodeSize} bytes). Verify this is correct.`
        );
      }

    } catch (error) {
      result.valid = false;
      result.errors.push(`Failed to validate shellcode: ${error instanceof Error ? error.message : String(error)}`);
      logger.debug('Shellcode validation error:', error);
    }

    return result;
  }

  /**
   * Determine if user confirmation is required for injection
   */
  requireConfirmation(
    targetValidation: TargetProcessValidation,
    payloadValidation: PayloadValidation
  ): ConfirmationRequirement {
    const result: ConfirmationRequirement = {
      required: false,
      overrideFlag: 'confirmed',
    };

    // DISABLED or PERMISSIVE: never require confirmation
    if (
      this.config.mode === InjectionValidationMode.DISABLED ||
      this.config.mode === InjectionValidationMode.PERMISSIVE
    ) {
      return result;
    }

    // BALANCED: never require confirmation (warnings only)
    if (this.config.mode === InjectionValidationMode.BALANCED) {
      return result;
    }

    // STRICT mode: require confirmation for unsigned processes or warnings
    if (this.config.mode === InjectionValidationMode.STRICT) {
      const reasons: string[] = [];

      if (targetValidation.isSigned === false) {
        reasons.push('target process is not digitally signed');
      }

      if (targetValidation.warnings.length > 0) {
        reasons.push('target process has validation warnings');
      }

      if (payloadValidation.warnings.length > 0) {
        reasons.push('payload has validation warnings');
      }

      if (reasons.length > 0) {
        result.required = true;
        result.reason = `Confirmation required: ${reasons.join(', ')}`;
      }
    }

    return result;
  }

  // ── Private helpers ──

  private computeFileHash(filePath: string): string {
    try {
      const fileBuffer = readFileSync(filePath);
      const hash = createHash('sha256');
      hash.update(fileBuffer);
      return hash.digest('hex');
    } catch (error) {
      logger.debug('Failed to compute file hash:', error);
      return '';
    }
  }

  private async checkWindowsSignature(executablePath?: string): Promise<{ isSigned: boolean; isValid: boolean }> {
    // Placeholder for Windows signature validation
    // Real implementation would use Get-AuthenticodeSignature or similar
    // For now, return unknown status
    return { isSigned: false, isValid: false };
  }
}

/**
 * Create validator instance from environment configuration
 */
export function createValidatorFromEnv(): InjectionValidator {
  const modeStr = (process.env.JSHOOK_INJECTION_VALIDATION_MODE ?? 'balanced').toLowerCase();
  const mode = Object.values(InjectionValidationMode).includes(modeStr as InjectionValidationMode)
    ? (modeStr as InjectionValidationMode)
    : InjectionValidationMode.BALANCED;

  const config: InjectionValidatorConfig = {
    mode,
    maxShellcodeSize: 1024 * 1024, // 1MB default
    requireHashForDll: mode === InjectionValidationMode.STRICT,
  };

  return new InjectionValidator(config);
}
