import type { MCPServerContext } from '@server/domains/shared/registry';
import type { ToolResponse } from '@server/domains/shared/ResponseBuilder';
import type { WebGPUDomainDependencies } from './types';
import {
  AdapterInfoHandler,
  ShaderCompileHandler,
  ShaderDisassembleHandler,
  TimingAnalysisHandler,
  MemoryLayoutHandler,
  CommandCaptureHandler,
} from './handlers/index.js';

/**
 * WebGPU domain handlers facade
 * Delegates to modular handler classes for each tool
 */
export class WebGPUHandlers {
  private adapterInfoHandler: AdapterInfoHandler;
  private shaderCompileHandler: ShaderCompileHandler;
  private shaderDisassembleHandler: ShaderDisassembleHandler;
  private timingAnalysisHandler: TimingAnalysisHandler;
  private memoryLayoutHandler: MemoryLayoutHandler;
  private commandCaptureHandler: CommandCaptureHandler;

  constructor(
    private ctx: MCPServerContext,
    deps?: WebGPUDomainDependencies
  ) {
    const d = deps ?? {
      pageController: ctx.pageController,
    };

    this.adapterInfoHandler = new AdapterInfoHandler(ctx, d);
    this.shaderCompileHandler = new ShaderCompileHandler(ctx, d);
    this.shaderDisassembleHandler = new ShaderDisassembleHandler(ctx, d);
    this.timingAnalysisHandler = new TimingAnalysisHandler(ctx, d);
    this.memoryLayoutHandler = new MemoryLayoutHandler(ctx, d);
    this.commandCaptureHandler = new CommandCaptureHandler(ctx, d);
  }

  async webgpu_adapter_info(args: Record<string, unknown>): Promise<ToolResponse> {
    return this.adapterInfoHandler.handle(args);
  }

  async webgpu_shader_compile(args: Record<string, unknown>): Promise<ToolResponse> {
    return this.shaderCompileHandler.handle(args);
  }

  async webgpu_shader_disassemble(args: Record<string, unknown>): Promise<ToolResponse> {
    return this.shaderDisassembleHandler.handle(args);
  }

  async webgpu_timing_analysis(args: Record<string, unknown>): Promise<ToolResponse> {
    return this.timingAnalysisHandler.handle(args);
  }

  async webgpu_memory_layout(args: Record<string, unknown>): Promise<ToolResponse> {
    return this.memoryLayoutHandler.handle(args);
  }

  async webgpu_capture_commands(args: Record<string, unknown>): Promise<ToolResponse> {
    return this.commandCaptureHandler.handle(args);
  }
}

export default WebGPUHandlers;
