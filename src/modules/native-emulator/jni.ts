/**
 * jni — JNIEnv/JavaVM emulation for the native emulator (A-plan / L4).
 *
 * This is the "native Android" core: a `.so`'s JNI entry points expect a
 * JNIEnv* whose every operation (FindClass, GetMethodID, GetStringUTFChars,
 * NewByteArray, …) dispatches through a function-pointer table. We materialise
 * that table in guest memory, back each implemented slot with a host stub, and
 * keep a host-side object table so opaque handles (jclass/jstring/jbyteArray/
 * jmethodID) map to real JS values.
 *
 * Memory model (double indirection, matching the real ABI):
 *   JNIEnv*  → [8-byte slot] → JNINativeInterface table (220 slots × 8 bytes)
 *   JavaVM*  → [8-byte slot] → JNIInvokeInterface table
 *
 * Function-table indices are the stable Oracle JNI ABI; only the slots we
 * implement are filled, the rest stay NULL (calling them would fault loudly,
 * which is the honest signal that we need to add one).
 */
import type { CpuEngine, HostContext } from './CpuEngine';
import { readGuestCString } from './c-strings';

export const JNI_VERSION_1_6 = 0x00010006;

/** JNINativeInterface slot indices (4 reserved slots precede GetVersion@4). */
export const JNI_INDEX = {
  GetVersion: 4,
  FindClass: 6,
  GetObjectClass: 31,
  IsInstanceOf: 32,
  GetMethodID: 33,
  CallObjectMethod: 34,
  CallBooleanMethod: 37,
  CallIntMethod: 49,
  CallLongMethod: 51,
  CallVoidMethod: 61,
  // Field access (instance).
  GetFieldID: 94,
  GetObjectField: 95,
  GetBooleanField: 96,
  GetIntField: 100,
  GetLongField: 101,
  SetObjectField: 104,
  SetBooleanField: 105,
  SetIntField: 109,
  SetLongField: 110,
  GetStaticMethodID: 113,
  CallStaticObjectMethod: 114,
  CallStaticBooleanMethod: 117,
  CallStaticIntMethod: 119,
  CallStaticLongMethod: 121,
  CallStaticVoidMethod: 143,
  // Static field access.
  GetStaticFieldID: 144,
  GetStaticObjectField: 145,
  GetStaticBooleanField: 146,
  GetStaticIntField: 150,
  GetStaticLongField: 151,
  SetStaticObjectField: 154,
  SetStaticIntField: 159,
  // Strings.
  NewStringUTF: 167,
  GetStringUTFLength: 168,
  GetStringUTFChars: 169,
  ReleaseStringUTFChars: 170,
  GetArrayLength: 171,
  // Object arrays.
  NewObjectArray: 172,
  GetObjectArrayElement: 173,
  SetObjectArrayElement: 174,
  NewByteArray: 176,
  GetByteArrayElements: 184,
  ReleaseByteArrayElements: 187,
  GetByteArrayRegion: 208,
  SetByteArrayRegion: 209,
  RegisterNatives: 215,
  GetJavaVM: 219,
  // Exceptions.
  Throw: 13,
  ThrowNew: 14,
  ExceptionOccurred: 15,
  ExceptionClear: 17,
  ExceptionCheck: 228,
  // References.
  NewGlobalRef: 21,
  DeleteGlobalRef: 22,
  DeleteLocalRef: 23,
  NewLocalRef: 25,
  IsSameObject: 24,
} as const;

/** JNIInvokeInterface (JavaVM) slot indices: 3 reserved, then the calls. */
export const JNI_INVOKE_INDEX = {
  DestroyJavaVM: 3,
  AttachCurrentThread: 4,
  DetachCurrentThread: 5,
  GetEnv: 6,
  AttachCurrentThreadAsDaemon: 7,
} as const;

const TABLE_SLOTS = 232; // ≥ highest index we touch (219) + headroom.
const POINTER_SIZE = 8;

// Guest memory layout for the JNI scaffolding (distinct high addresses).
const ENV_PTR_ADDR = 0x6000_0000; // holds the table base (what JNIEnv* points at)
const ENV_TABLE_ADDR = 0x6000_0100; // JNINativeInterface table base
const STUB_BASE = 0x6010_0000; // unique guest addr per implemented function stub
const VM_PTR_ADDR = 0x6002_0000; // holds the invoke-table base (what JavaVM* points at)
const VM_TABLE_ADDR = 0x6002_0100; // JNIInvokeInterface table base
const VM_STUB_BASE = 0x6012_0000;

// Host-side handle space (opaque jobject/jclass/jstring/jarray values).
const HANDLE_BASE = 0x7000_0000;

interface JavaClass {
  name: string;
  /** methodName+signature → jmethodID handle. */
  methods: Map<string, number>;
  /** fieldName+signature → jfieldID handle. */
  fields: Map<string, number>;
}

/** A native method registered via RegisterNatives (or installed directly). */
export interface NativeMethodBinding {
  name: string;
  signature: string;
  /** Guest address of the native implementation (entry to BL/callSymbol). */
  fnAddr: number;
}

/** A mock Java field's declared value (the constant native code reads back). */
interface JavaFieldEntry {
  className: string;
  name: string;
  sig: string;
  /** Declared value: a primitive bigint, or a handle (string/bytes) allocated lazily. */
  value: bigint;
}

export class JniEnvironment {
  private readonly engine: CpuEngine;
  private stubBump = STUB_BASE;
  private vmStubBump = VM_STUB_BASE;
  private handleBump = HANDLE_BASE;

  /** handle → host value (class/string/byte-array/etc.). */
  private readonly handles = new Map<number, unknown>();
  private readonly classes = new Map<string, number>(); // name → jclass handle
  private readonly classByHandle = new Map<number, JavaClass>();
  /** "className#method#sig" → fnAddr, populated by RegisterNatives. */
  private readonly natives = new Map<string, NativeMethodBinding>();
  /** Live GetByteArrayElements pointers → owning array handle, for write-back on Release. */
  private readonly arrayElements = new Map<number, { handle: number; length: number }>();
  /** Mock "Java world": jmethodID handle → its JS implementation. */
  private readonly javaMethods = new Map<number, JavaMethodEntry>();
  /** Mock Java fields: jfieldID handle → its declared value. */
  private readonly javaFields = new Map<number, JavaFieldEntry>();
  /** Currently-pending exception handle (0 = none), set by Throw/ThrowNew. */
  private pendingException = 0;

  constructor(engine: CpuEngine) {
    this.engine = engine;
    this.installEnvTable();
    this.installVmTable();
  }

  /** The guest JNIEnv* value to pass as the first arg of a Java_* function. */
  envPointer(): number {
    return ENV_PTR_ADDR;
  }

  /** The guest JavaVM* value to pass to JNI_OnLoad. */
  javaVmPointer(): number {
    return VM_PTR_ADDR;
  }

  /** Pre-register a class so FindClass resolves it; returns its jclass handle. */
  defineClass(name: string): number {
    const existing = this.classes.get(name);
    if (existing !== undefined) return existing;
    const handle = this.allocHandle({ kind: 'class', name });
    this.classes.set(name, handle);
    this.classByHandle.set(handle, { name, methods: new Map(), fields: new Map() });
    return handle;
  }

  /** Resolve a jclass handle back to its class name (host-side introspection). */
  classNameOf(handle: number): string | undefined {
    return this.classByHandle.get(handle)?.name;
  }

  /**
   * Register a mock Java method implementation. When emulated native code calls
   * GetMethodID/GetStaticMethodID for this class+name+sig and then a Call*Method
   * through the returned jmethodID, the dispatch lands in `impl` — a programmable
   * "Java world" so a native routine can call back up into Java (e.g. to fetch a
   * value it then encrypts). `impl` receives the Java arguments (x3.. as bigint)
   * and the receiver object handle; its return becomes the Call*Method result.
   */
  registerJavaMethod(className: string, name: string, sig: string, impl: JavaMethodImpl): void {
    const cls = this.classByHandle.get(this.defineClass(className));
    if (!cls) return;
    const key = `${name}#${sig}`;
    let id = cls.methods.get(key);
    if (id === undefined) {
      id = this.allocHandle({ kind: 'method', name, sig, cls: className });
      cls.methods.set(key, id);
    }
    this.javaMethods.set(id, { className, name, sig, impl });
  }

  /**
   * Register a mock Java field. When emulated native code calls
   * GetFieldID/GetStaticFieldID then Get<Type>Field, the dispatch returns this
   * declared `value` (a primitive, or a handle for object fields). Mirrors
   * registerJavaMethod for the "Java world" a native routine reads constants from.
   */
  registerJavaField(className: string, name: string, sig: string, value: bigint): void {
    const cls = this.classByHandle.get(this.defineClass(className));
    if (!cls) return;
    const key = `${name}#${sig}`;
    let id = cls.fields.get(key);
    if (id === undefined) {
      id = this.allocHandle({ kind: 'field', name, sig, cls: className });
      cls.fields.set(key, id);
    }
    this.javaFields.set(id, { className, name, sig, value });
  }

  /** Look up a native binding registered for a class/method/signature. */
  nativeBinding(className: string, method: string, sig: string): NativeMethodBinding | undefined {
    return this.natives.get(`${className}#${method}#${sig}`);
  }

  /** Read a host value previously stored behind a handle. */
  valueOf(handle: number): unknown {
    return this.handles.get(handle);
  }

  /** Allocate a fresh opaque handle bound to a host value. */
  allocHandle(value: unknown): number {
    const handle = this.handleBump;
    this.handleBump += POINTER_SIZE;
    this.handles.set(handle, value);
    return handle;
  }

  /**
   * Release all JNI resources: object handles (jclass/jstring/jbyteArray),
   * class registry, method/field registrations, native bindings, and
   * GetByteArrayElements tracking.
   *
   * Idempotent: safe to call multiple times. Follows the disposal pattern for
   * emulator-backed JNI environments, ensuring no handle leaks accumulate across
   * repeated session create/destroy cycles.
   */
  dispose(): void {
    // Clear handle table (releases all jclass/jstring/jbyteArray/jmethodID/jfieldID)
    this.handles.clear();
    this.handleBump = HANDLE_BASE;

    // Clear class registry
    this.classes.clear();
    this.classByHandle.clear();

    // Clear native method bindings
    this.natives.clear();

    // Clear GetByteArrayElements tracking
    this.arrayElements.clear();

    // Clear mock Java methods and fields
    this.javaMethods.clear();
    this.javaFields.clear();

    // Reset stub allocators
    this.stubBump = STUB_BASE;
    this.vmStubBump = VM_STUB_BASE;

    // Clear pending exception
    this.pendingException = 0;
  }

  // ── JNINativeInterface table construction ──

  private installEnvTable(): void {
    this.engine.mapMemory(ENV_PTR_ADDR, POINTER_SIZE);
    this.engine.mapMemory(ENV_TABLE_ADDR, TABLE_SLOTS * POINTER_SIZE);
    this.writePointer(ENV_PTR_ADDR, ENV_TABLE_ADDR); // *JNIEnv = table base

    this.bind(JNI_INDEX.GetVersion, () => BigInt(JNI_VERSION_1_6));
    this.bind(JNI_INDEX.FindClass, (ctx) => this.jniFindClass(ctx));
    this.bind(JNI_INDEX.GetMethodID, (ctx) => this.jniGetMethodID(ctx));
    this.bind(JNI_INDEX.RegisterNatives, (ctx) => this.jniRegisterNatives(ctx));
    this.bind(JNI_INDEX.NewStringUTF, (ctx) => this.jniNewStringUTF(ctx));
    this.bind(JNI_INDEX.GetStringUTFChars, (ctx) => this.jniGetStringUTFChars(ctx));
    this.bind(JNI_INDEX.ReleaseStringUTFChars, () => undefined);
    this.bind(JNI_INDEX.NewByteArray, (ctx) => this.jniNewByteArray(ctx));
    this.bind(JNI_INDEX.GetArrayLength, (ctx) => this.jniGetArrayLength(ctx));
    this.bind(JNI_INDEX.GetByteArrayElements, (ctx) => this.jniGetByteArrayElements(ctx));
    this.bind(JNI_INDEX.ReleaseByteArrayElements, (ctx) => this.jniReleaseByteArrayElements(ctx));
    this.bind(JNI_INDEX.SetByteArrayRegion, (ctx) => this.jniSetByteArrayRegion(ctx));
    this.bind(JNI_INDEX.GetByteArrayRegion, (ctx) => this.jniGetByteArrayRegion(ctx));
    this.bind(JNI_INDEX.GetJavaVM, (ctx) => this.jniGetJavaVM(ctx));
    // Call*Method family + static method lookup — the reflection callback path.
    this.bind(JNI_INDEX.GetStaticMethodID, (ctx) => this.jniGetMethodID(ctx));
    this.bind(JNI_INDEX.CallObjectMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallBooleanMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallIntMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallVoidMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallStaticObjectMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallStaticIntMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallLongMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallStaticBooleanMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallStaticLongMethod, (ctx) => this.jniCallMethod(ctx));
    this.bind(JNI_INDEX.CallStaticVoidMethod, (ctx) => this.jniCallMethod(ctx));

    // Field access — GetFieldID/GetStaticFieldID return a jfieldID; the typed
    // getters return the declared mock value. Setters are accepted as no-ops
    // (the mock "Java world" is read-only from native code's perspective).
    this.bind(JNI_INDEX.GetFieldID, (ctx) => this.jniGetFieldID(ctx));
    this.bind(JNI_INDEX.GetStaticFieldID, (ctx) => this.jniGetFieldID(ctx));
    this.bind(JNI_INDEX.GetObjectField, (ctx) => this.jniGetField(ctx));
    this.bind(JNI_INDEX.GetBooleanField, (ctx) => this.jniGetField(ctx));
    this.bind(JNI_INDEX.GetIntField, (ctx) => this.jniGetField(ctx));
    this.bind(JNI_INDEX.GetLongField, (ctx) => this.jniGetField(ctx));
    this.bind(JNI_INDEX.GetStaticObjectField, (ctx) => this.jniGetStaticField(ctx));
    this.bind(JNI_INDEX.GetStaticBooleanField, (ctx) => this.jniGetStaticField(ctx));
    this.bind(JNI_INDEX.GetStaticIntField, (ctx) => this.jniGetStaticField(ctx));
    this.bind(JNI_INDEX.GetStaticLongField, (ctx) => this.jniGetStaticField(ctx));
    this.bind(JNI_INDEX.SetObjectField, () => undefined);
    this.bind(JNI_INDEX.SetBooleanField, () => undefined);
    this.bind(JNI_INDEX.SetIntField, () => undefined);
    this.bind(JNI_INDEX.SetLongField, () => undefined);
    this.bind(JNI_INDEX.SetStaticObjectField, () => undefined);
    this.bind(JNI_INDEX.SetStaticIntField, () => undefined);

    // Strings & arrays beyond the byte-array core.
    this.bind(JNI_INDEX.GetStringUTFLength, (ctx) => this.jniGetStringUTFLength(ctx));
    this.bind(JNI_INDEX.GetObjectClass, (ctx) => this.jniGetObjectClass(ctx));
    this.bind(JNI_INDEX.NewObjectArray, (ctx) => this.jniNewObjectArray(ctx));
    this.bind(JNI_INDEX.GetObjectArrayElement, (ctx) => this.jniGetObjectArrayElement(ctx));
    this.bind(JNI_INDEX.SetObjectArrayElement, (ctx) => this.jniSetObjectArrayElement(ctx));

    // Exceptions — a minimal model: Throw/ThrowNew record a pending handle that
    // ExceptionCheck/ExceptionOccurred report and ExceptionClear resets.
    this.bind(JNI_INDEX.Throw, (ctx) => {
      this.pendingException = Number(ctx.x(1));
      return 0n;
    });
    this.bind(JNI_INDEX.ThrowNew, (ctx) => {
      this.pendingException = this.allocHandle({ kind: 'throwable', cls: Number(ctx.x(1)) });
      return 0n;
    });
    this.bind(JNI_INDEX.ExceptionOccurred, () => BigInt(this.pendingException));
    this.bind(JNI_INDEX.ExceptionCheck, () => (this.pendingException !== 0 ? 1n : 0n));
    this.bind(JNI_INDEX.ExceptionClear, () => {
      this.pendingException = 0;
      return undefined;
    });

    // References — handles are process-global in this model, so global/local ref
    // management is identity (return the same handle) and deletion is a no-op.
    this.bind(JNI_INDEX.NewGlobalRef, (ctx) => ctx.x(1));
    this.bind(JNI_INDEX.NewLocalRef, (ctx) => ctx.x(1));
    this.bind(JNI_INDEX.DeleteGlobalRef, () => undefined);
    this.bind(JNI_INDEX.DeleteLocalRef, () => undefined);
    this.bind(JNI_INDEX.IsSameObject, (ctx) => (ctx.x(1) === ctx.x(2) ? 1n : 0n));
    this.bind(JNI_INDEX.IsInstanceOf, () => 1n); // optimistic: assume instance-of holds
  }

  private installVmTable(): void {
    this.engine.mapMemory(VM_PTR_ADDR, POINTER_SIZE);
    this.engine.mapMemory(VM_TABLE_ADDR, 16 * POINTER_SIZE);
    this.writePointer(VM_PTR_ADDR, VM_TABLE_ADDR);
    // GetEnv(vm, void** out, version): store the JNIEnv*, return 0 (JNI_OK).
    this.bindVm(JNI_INVOKE_INDEX.GetEnv, (ctx) => {
      const out = Number(ctx.x(1));
      this.writePointer(out, ENV_PTR_ADDR);
      return 0n;
    });
  }

  /** Bind a JNINativeInterface slot to a host stub and write its addr into the table. */
  private bind(index: number, fn: (ctx: HostContext) => bigint | number | void): void {
    const stubAddr = this.stubBump;
    this.stubBump += POINTER_SIZE;
    this.engine.registerHostFunction(stubAddr, fn);
    this.writePointer(ENV_TABLE_ADDR + index * POINTER_SIZE, stubAddr);
  }

  private bindVm(index: number, fn: (ctx: HostContext) => bigint | number | void): void {
    const stubAddr = this.vmStubBump;
    this.vmStubBump += POINTER_SIZE;
    this.engine.registerHostFunction(stubAddr, fn);
    this.writePointer(VM_TABLE_ADDR + index * POINTER_SIZE, stubAddr);
  }

  // ── JNI function implementations ──

  /** jclass FindClass(JNIEnv*, const char* name): x1 = name. */
  private jniFindClass(ctx: HostContext): bigint {
    const name = this.readCString(ctx, Number(ctx.x(1)));
    return BigInt(this.defineClass(name)); // auto-define unknown classes
  }

  /** jmethodID GetMethodID(JNIEnv*, jclass, const char* name, const char* sig). */
  private jniGetMethodID(ctx: HostContext): bigint {
    const cls = this.classByHandle.get(Number(ctx.x(1)));
    const name = this.readCString(ctx, Number(ctx.x(2)));
    const sig = this.readCString(ctx, Number(ctx.x(3)));
    const key = `${name}#${sig}`;
    if (cls) {
      const existing = cls.methods.get(key);
      if (existing !== undefined) return BigInt(existing);
      const id = this.allocHandle({ kind: 'method', name, sig, cls: cls.name });
      cls.methods.set(key, id);
      return BigInt(id);
    }
    return BigInt(this.allocHandle({ kind: 'method', name, sig }));
  }

  /**
   * jint RegisterNatives(JNIEnv*, jclass, const JNINativeMethod* methods, jint n).
   * JNINativeMethod = { char* name; char* signature; void* fnPtr } (24 bytes).
   */
  private jniRegisterNatives(ctx: HostContext): bigint {
    const cls = this.classByHandle.get(Number(ctx.x(1)));
    const methods = Number(ctx.x(2));
    const count = Number(ctx.x(3));
    for (let i = 0; i < count; i++) {
      const rec = methods + i * 24;
      const namePtr = this.readPointer(ctx, rec);
      const sigPtr = this.readPointer(ctx, rec + 8);
      const fnAddr = this.readPointer(ctx, rec + 16);
      const name = this.readCString(ctx, namePtr);
      const signature = this.readCString(ctx, sigPtr);
      const className = cls?.name ?? '';
      this.natives.set(`${className}#${name}#${signature}`, { name, signature, fnAddr });
    }
    return 0n; // JNI_OK
  }

  /** jstring NewStringUTF(JNIEnv*, const char* bytes): x1 = bytes. */
  private jniNewStringUTF(ctx: HostContext): bigint {
    const str = this.readCString(ctx, Number(ctx.x(1)));
    return BigInt(this.allocHandle({ kind: 'string', value: str }));
  }

  /** const char* GetStringUTFChars(JNIEnv*, jstring, jboolean* isCopy). */
  private jniGetStringUTFChars(ctx: HostContext): bigint {
    const value = this.handles.get(Number(ctx.x(1)));
    const str = isStringValue(value) ? value.value : '';
    const bytes = new TextEncoder().encode(str + '\0');
    const addr = this.allocGuestBuffer(bytes);
    return BigInt(addr);
  }

  /** jbyteArray NewByteArray(JNIEnv*, jsize length): x1 = length. */
  private jniNewByteArray(ctx: HostContext): bigint {
    const length = Number(ctx.x(1));
    return BigInt(this.allocHandle({ kind: 'bytes', value: new Uint8Array(length) }));
  }

  /**
   * jsize GetArrayLength(JNIEnv*, jarray). Works for both jbyteArray and any
   * object array (String[]/Object[]) — a native loop like RootBeer's
   * `for (i=0; i<GetArrayLength(paths); i++)` drives off this count, so an
   * object array must report its real length, not 0.
   */
  private jniGetArrayLength(ctx: HostContext): bigint {
    const value = this.handles.get(Number(ctx.x(1)));
    if (isBytesValue(value)) return BigInt(value.value.length);
    if (isObjArrayValue(value)) return BigInt(value.value.length);
    return 0n;
  }

  /** jbyte* GetByteArrayElements(JNIEnv*, jbyteArray, jboolean* isCopy). */
  private jniGetByteArrayElements(ctx: HostContext): bigint {
    const handle = Number(ctx.x(1));
    const value = this.handles.get(handle);
    const bytes = isBytesValue(value) ? value.value : new Uint8Array(0);
    const addr = this.allocGuestBuffer(bytes);
    // Track the live pointer so ReleaseByteArrayElements can copy edits back to
    // the array handle — matching real JNI, where mode 0 commits and frees.
    this.arrayElements.set(addr, { handle, length: bytes.length });
    return BigInt(addr);
  }

  /** void ReleaseByteArrayElements(JNIEnv*, jbyteArray, jbyte* elems, jint mode). */
  private jniReleaseByteArrayElements(ctx: HostContext): void {
    const elems = Number(ctx.x(2));
    const mode = Number(ctx.x(3));
    const tracked = this.arrayElements.get(elems);
    if (!tracked) return;
    const value = this.handles.get(tracked.handle);
    // mode 0 (commit + free) and JNI_COMMIT (1) write edits back to the array.
    if (mode !== 2 /* JNI_ABORT */ && isBytesValue(value)) {
      value.value.set(ctx.read(elems, tracked.length));
    }
    if (mode !== 1 /* JNI_COMMIT keeps the buffer */) this.arrayElements.delete(elems);
  }

  /** void SetByteArrayRegion(JNIEnv*, jbyteArray, jsize start, jsize len, jbyte* buf). */
  private jniSetByteArrayRegion(ctx: HostContext): void {
    const value = this.handles.get(Number(ctx.x(1)));
    if (!isBytesValue(value)) return;
    const start = Number(ctx.x(2));
    const len = Number(ctx.x(3));
    const buf = Number(ctx.x(4));
    const src = ctx.read(buf, len);
    value.value.set(src.subarray(0, len), start);
  }

  /** void GetByteArrayRegion(JNIEnv*, jbyteArray, jsize start, jsize len, jbyte* buf). */
  private jniGetByteArrayRegion(ctx: HostContext): void {
    const value = this.handles.get(Number(ctx.x(1)));
    if (!isBytesValue(value)) return;
    const start = Number(ctx.x(2));
    const len = Number(ctx.x(3));
    const buf = Number(ctx.x(4));
    ctx.write(buf, value.value.subarray(start, start + len));
  }

  /** jint GetJavaVM(JNIEnv*, JavaVM** vm): store the VM pointer, return 0. */
  private jniGetJavaVM(ctx: HostContext): bigint {
    const out = Number(ctx.x(1));
    this.writePointer(out, VM_PTR_ADDR);
    return 0n;
  }

  /**
   * Call*Method dispatch: x1 = receiver (jobject/jclass), x2 = jmethodID,
   * x3..x7 = up to five Java arguments. Routes to the registered mock impl and
   * returns whatever it produces in x0. Unregistered methods return 0 (a benign
   * null/zero), which keeps a partially-modelled Java world from hard-faulting.
   */
  private jniCallMethod(ctx: HostContext): bigint {
    const self = Number(ctx.x(1));
    const methodId = Number(ctx.x(2));
    const entry = this.javaMethods.get(methodId);
    if (!entry) return 0n;
    const args = [ctx.x(3), ctx.x(4), ctx.x(5), ctx.x(6), ctx.x(7)];
    const result = entry.impl({ args, self, jni: this });
    return result === undefined ? 0n : BigInt.asUintN(64, BigInt(result));
  }

  /**
   * jfieldID GetFieldID/GetStaticFieldID(JNIEnv*, jclass, name, sig): resolve (or
   * lazily mint) the field handle for the class so a later Get*Field can find it.
   */
  private jniGetFieldID(ctx: HostContext): bigint {
    const cls = this.classByHandle.get(Number(ctx.x(1)));
    const name = this.readCString(ctx, Number(ctx.x(2)));
    const sig = this.readCString(ctx, Number(ctx.x(3)));
    const key = `${name}#${sig}`;
    if (cls) {
      const existing = cls.fields.get(key);
      if (existing !== undefined) return BigInt(existing);
      const id = this.allocHandle({ kind: 'field', name, sig, cls: cls.name });
      cls.fields.set(key, id);
      return BigInt(id);
    }
    return BigInt(this.allocHandle({ kind: 'field', name, sig }));
  }

  /** Get<Type>Field(JNIEnv*, jobject, jfieldID): return the declared mock value. */
  private jniGetField(ctx: HostContext): bigint {
    const entry = this.javaFields.get(Number(ctx.x(2)));
    return entry ? entry.value : 0n;
  }

  /** GetStatic<Type>Field(JNIEnv*, jclass, jfieldID): same lookup as instance. */
  private jniGetStaticField(ctx: HostContext): bigint {
    const entry = this.javaFields.get(Number(ctx.x(2)));
    return entry ? entry.value : 0n;
  }

  /** jsize GetStringUTFLength(JNIEnv*, jstring): UTF-8 byte length of the string. */
  private jniGetStringUTFLength(ctx: HostContext): bigint {
    const value = this.handles.get(Number(ctx.x(1)));
    const str = isStringValue(value) ? value.value : '';
    return BigInt(new TextEncoder().encode(str).length);
  }

  /** jclass GetObjectClass(JNIEnv*, jobject): the object's class handle (best-effort). */
  private jniGetObjectClass(ctx: HostContext): bigint {
    const value = this.handles.get(Number(ctx.x(1)));
    if (value && typeof value === 'object' && 'cls' in value) {
      const clsName = (value as { cls?: string }).cls;
      if (typeof clsName === 'string') return BigInt(this.defineClass(clsName));
    }
    return BigInt(this.defineClass('java/lang/Object'));
  }

  /** jobjectArray NewObjectArray(JNIEnv*, jsize len, jclass, jobject init). */
  private jniNewObjectArray(ctx: HostContext): bigint {
    const length = Number(ctx.x(1));
    const init = ctx.x(3);
    const arr = Array.from<bigint>({ length }).fill(init);
    return BigInt(this.allocHandle({ kind: 'objarray', value: arr }));
  }

  /** jobject GetObjectArrayElement(JNIEnv*, jobjectArray, jsize index). */
  private jniGetObjectArrayElement(ctx: HostContext): bigint {
    const value = this.handles.get(Number(ctx.x(1)));
    const idx = Number(ctx.x(2));
    if (isObjArrayValue(value)) return value.value[idx] ?? 0n;
    return 0n;
  }

  /** void SetObjectArrayElement(JNIEnv*, jobjectArray, jsize index, jobject val). */
  private jniSetObjectArrayElement(ctx: HostContext): void {
    const value = this.handles.get(Number(ctx.x(1)));
    const idx = Number(ctx.x(2));
    if (isObjArrayValue(value)) value.value[idx] = ctx.x(3);
  }

  // ── Guest memory helpers ──

  /** Map a fresh guest buffer, copy bytes in, return its address. */
  private allocGuestBuffer(bytes: Uint8Array): number {
    const addr = this.handleBump;
    this.handleBump += Math.max(POINTER_SIZE, bytes.length + 8);
    this.engine.mapMemory(addr, Math.max(POINTER_SIZE, bytes.length + 8));
    if (bytes.length > 0) this.engine.writeCode(addr, bytes);
    return addr;
  }

  private writePointer(addr: number, value: number): void {
    const bytes = new Uint8Array(POINTER_SIZE);
    let v = BigInt(value);
    for (let i = 0; i < POINTER_SIZE; i++) {
      bytes[i] = Number(v & 0xffn);
      v >>= 8n;
    }
    this.engine.writeCode(addr, bytes);
  }

  private readPointer(ctx: HostContext, addr: number): number {
    const bytes = ctx.read(addr, POINTER_SIZE);
    let value = 0;
    for (let i = 0; i < POINTER_SIZE; i++) value += bytes[i]! * 2 ** (i * 8);
    return value;
  }

  private readCString(ctx: HostContext, addr: number): string {
    return readGuestCString(ctx, addr);
  }
}

interface StringValue {
  kind: 'string';
  value: string;
}
interface BytesValue {
  kind: 'bytes';
  value: Uint8Array;
}
interface ObjArrayValue {
  kind: 'objarray';
  value: bigint[];
}

/** Arguments handed to a mock Java method implementation. */
export interface JavaMethodCall {
  /** Java arguments as passed in x3..x7 (BigInt, 64-bit). */
  args: bigint[];
  /** The receiver: jobject handle (instance calls) or jclass handle (static). */
  self: number;
  /** The owning environment, for allocating return handles (strings/arrays). */
  jni: JniEnvironment;
}

/** A mock Java method: returns the Call*Method result (handle/int/bool) or void. */
export type JavaMethodImpl = (call: JavaMethodCall) => bigint | number | void;

interface JavaMethodEntry {
  className: string;
  name: string;
  sig: string;
  impl: JavaMethodImpl;
}

function isStringValue(v: unknown): v is StringValue {
  return typeof v === 'object' && v !== null && (v as { kind?: string }).kind === 'string';
}

function isBytesValue(v: unknown): v is BytesValue {
  return typeof v === 'object' && v !== null && (v as { kind?: string }).kind === 'bytes';
}

function isObjArrayValue(v: unknown): v is ObjArrayValue {
  return typeof v === 'object' && v !== null && (v as { kind?: string }).kind === 'objarray';
}
