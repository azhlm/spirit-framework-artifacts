"""
Stage 1 — Junk Array + Null Variable Deobfuscator (Binary Ninja Plugin)

Removes two complementary obfuscation patterns found in the Stage 1 loader:

  1. **Junk Array (RDX/arg2)** — The obfuscator passes a pointer to a
     scratch buffer ("junk array") as the second argument.  Instructions
     that only read/write this tainted buffer have no semantic effect.
     Detected via Triton's taint propagation engine; highlighted in red.

  2. **Null Variable (RCX/arg1)** — The first argument is always zero.
     Arithmetic instructions that use this null value as a source-only
     operand (e.g., `add r10, rcx` where RCX=0) are identity operations.
     Detected via manual null-propagation tracking; highlighted in orange.

Usage (Binary Ninja Script Console):
    from stage1_junk_array_and_null_variable import JunkArrayDeobfuscator

    d = JunkArrayDeobfuscator(bv, bv.get_function_at(0x...))
    d.run()        # highlight junk instructions
    d.patch()      # NOP-patch highlighted instructions

    # Process all functions
    for func in bv.functions:
        JunkArrayDeobfuscator(bv, func).run()

Environment: Binary Ninja with the `triton` package installed.
"""

from triton import (
    TritonContext, ARCH, MODE, AST_REPRESENTATION,
    Instruction, MemoryAccess,
)
from binaryninja import (
    HighlightStandardColor,
    log_info,
)


# ======================================================================
#  Constants
# ======================================================================

JUNKARRAY_BASE = 0xDEAD0000
JUNKARRAY_SIZE = 0x1000
STACK_BASE     = 0x7FFFFFFFE000

FLAG_REGISTERS = frozenset({
    "eflags", "af", "cf", "df", "if", "of", "pf", "sf", "tf", "zf",
})

NEVER_NOP_MNEMONICS = frozenset({
    "ret", "retn", "syscall", "int", "int3",
    "leave", "push", "pop", "nop",
})

CONTROL_FLOW_MNEMONICS = frozenset({
    "je", "jne", "jz", "jnz", "ja", "jae", "jb", "jbe",
    "jg", "jge", "jl", "jle", "js", "jns", "jo", "jno",
    "jp", "jnp", "jcxz", "jecxz", "jrcxz", "jmp",
})

VOLATILE_REGS = ["rax", "rcx", "rdx", "r8", "r9", "r10", "r11"]

# MOV-like mnemonics: transfer data without computation.
# Null values propagate through these unchanged.
MOV_MNEMONICS = frozenset({
    "mov", "movzx", "movsx", "movsxd", "movabs",
    "cmova", "cmovae", "cmovb", "cmovbe", "cmovc",
    "cmove", "cmovg", "cmovge", "cmovl", "cmovle",
    "cmovna", "cmovnae", "cmovnb", "cmovnbe", "cmovnc",
    "cmovne", "cmovng", "cmovnge", "cmovnl", "cmovnle",
    "cmovno", "cmovnp", "cmovns", "cmovnz", "cmovo",
    "cmovp", "cmovs", "cmovz",
})

# Arithmetic operations where a null (0) source operand makes the
# instruction a no-op: add dst,0 = dst; sub dst,0 = dst; etc.
NULL_IDENTITY_OPS = frozenset({
    "add", "sub", "or", "xor", "adc", "sbb",
})


# ======================================================================
#  Register parent mapping (sub-register -> 64-bit parent)
# ======================================================================

_REG_PARENT = {}
for _p, _children in {
    "rax": ["rax", "eax", "ax", "al", "ah"],
    "rbx": ["rbx", "ebx", "bx", "bl", "bh"],
    "rcx": ["rcx", "ecx", "cx", "cl", "ch"],
    "rdx": ["rdx", "edx", "dx", "dl", "dh"],
    "rsi": ["rsi", "esi", "si", "sil"],
    "rdi": ["rdi", "edi", "di", "dil"],
    "rbp": ["rbp", "ebp", "bp", "bpl"],
    "rsp": ["rsp", "esp", "sp", "spl"],
    "r8":  ["r8", "r8d", "r8w", "r8b"],
    "r9":  ["r9", "r9d", "r9w", "r9b"],
    "r10": ["r10", "r10d", "r10w", "r10b"],
    "r11": ["r11", "r11d", "r11w", "r11b"],
    "r12": ["r12", "r12d", "r12w", "r12b"],
    "r13": ["r13", "r13d", "r13w", "r13b"],
    "r14": ["r14", "r14d", "r14w", "r14b"],
    "r15": ["r15", "r15d", "r15w", "r15b"],
}.items():
    for _c in _children:
        _REG_PARENT[_c] = _p


# ======================================================================
#  Main class
# ======================================================================

class JunkArrayDeobfuscator:

    def __init__(self, bv, func):
        self.bv = bv
        self.func = func
        self.junk_addrs = []          # Junk array taint: red highlight
        self.null_addrs = []          # Null no-op arithmetic: orange highlight
        self.ctx = self._setup()

        # Null-propagation state (tracked independently of Triton taint)
        self._null_regs = {"rcx"}     # RCX = arg1 = 0 at function entry
        self._null_mems = set()       # Concrete stack addresses holding null

    # ------------------------------------------------------------------
    #  Triton setup
    # ------------------------------------------------------------------

    def _setup(self):
        ctx = TritonContext(ARCH.X86_64)
        ctx.setMode(MODE.ALIGNED_MEMORY, True)
        ctx.setMode(MODE.TAINT_THROUGH_POINTERS, True)
        ctx.setAstRepresentationMode(AST_REPRESENTATION.PYTHON)

        ctx.setConcreteRegisterValue(ctx.registers.rsp, STACK_BASE)
        ctx.setConcreteRegisterValue(ctx.registers.rbp, STACK_BASE)

        # arg1 (RCX) = 0 (null variable)
        ctx.setConcreteRegisterValue(ctx.registers.rcx, 0x0)
        # arg2 (RDX) = junk array base (tainted)
        ctx.setConcreteRegisterValue(ctx.registers.rdx, JUNKARRAY_BASE)
        ctx.taintRegister(ctx.registers.rdx)

        # Taint the shadow-space copy of RDX
        shadow_rdx = MemoryAccess(STACK_BASE + 0x10, 8)
        ctx.setConcreteMemoryValue(shadow_rdx, JUNKARRAY_BASE)
        ctx.taintMemory(shadow_rdx)

        # Taint the entire junk array region
        for offset in range(0, JUNKARRAY_SIZE, 4):
            mem = MemoryAccess(JUNKARRAY_BASE + offset, 4)
            ctx.setConcreteMemoryValue(mem, 0)
            ctx.taintMemory(mem)

        return ctx

    # ------------------------------------------------------------------
    #  Register parent helper
    # ------------------------------------------------------------------

    def _parent(self, name):
        return _REG_PARENT.get(name.lower(), name.lower())

    def _read_parents(self, inst):
        return {
            self._parent(r.getName())
            for r, _ in inst.getReadRegisters()
            if r.getName().lower() not in FLAG_REGISTERS
        }

    def _write_parents(self, inst):
        return {
            self._parent(r.getName())
            for r, _ in inst.getWrittenRegisters()
            if r.getName().lower() not in FLAG_REGISTERS
        }

    # ------------------------------------------------------------------
    #  Null-propagation tracking
    # ------------------------------------------------------------------

    def _is_null_junk(self, inst):
        """
        Determine whether this instruction is a no-op due to a null
        source operand.  Example: `add r10, rdx_null` => r10 += 0 => no-op.
        Only returns True when the null value is source-only (read but not
        written), so `add rdx_null, r10` (which overwrites rdx) is NOT junk.
        """
        mnem = inst.getDisassembly().lower().split()[0]
        if mnem not in NULL_IDENTITY_OPS:
            return False

        reads = self._read_parents(inst)
        writes = self._write_parents(inst)
        source_only = reads - writes

        if source_only & self._null_regs:
            return True

        # Check source-only memory loads against known null stack slots
        load_addrs = {m.getAddress() for m, _ in inst.getLoadAccess()}
        store_addrs = {m.getAddress() for m, _ in inst.getStoreAccess()}
        if (load_addrs - store_addrs) & self._null_mems:
            return True

        return False

    def _update_null(self, inst):
        """
        Update null-propagation state after processing an instruction.

        Rules:
          - mov dst, null_src    => dst becomes null
          - mov dst, real_src    => dst loses null status
          - xor/sub reg, reg     => reg = 0 => becomes null
          - other arithmetic     => written outputs lose null status
        """
        mnem = inst.getDisassembly().lower().split()[0]
        reads = self._read_parents(inst)
        writes = self._write_parents(inst)

        # xor reg, reg / sub reg, reg => result is zero => null
        if mnem in ("xor", "sub") and reads == writes and len(reads) == 1:
            self._null_regs.update(writes)
            return

        # MOV-like: propagate or clear null status
        if mnem in MOV_MNEMONICS:
            has_null_src = bool((reads - writes) & self._null_regs)
            if not has_null_src:
                for m, _ in inst.getLoadAccess():
                    if m.getAddress() in self._null_mems:
                        has_null_src = True
                        break

            for w in writes:
                if w == "rsp":
                    continue
                if has_null_src:
                    self._null_regs.add(w)
                else:
                    self._null_regs.discard(w)

            for m, _ in inst.getStoreAccess():
                addr = m.getAddress()
                if reads & self._null_regs:
                    self._null_mems.add(addr)
                else:
                    self._null_mems.discard(addr)
            return

        # Non-MOV: written outputs lose null status
        for w in writes:
            self._null_regs.discard(w)
        for m, _ in inst.getStoreAccess():
            self._null_mems.discard(m.getAddress())

    # ------------------------------------------------------------------
    #  Junk array classification (Triton taint)
    # ------------------------------------------------------------------

    def _classify(self, inst):
        """
        Classify an instruction based on Triton taint analysis.

        Returns one of:
          CALL  — function call (handled separately)
          SKIP  — must not be patched (ret, push, pop, etc.)
          JUNK  — all outputs are tainted (pure junk)
          MIXED — some outputs tainted (treated as junk)
          SUSPECT — tainted conditional jump
          CLEAN — no taint involvement
        """
        disasm = inst.getDisassembly().lower()
        mnem = disasm.split()[0] if disasm else ""

        if mnem == "call":
            return "CALL"
        if mnem in NEVER_NOP_MNEMONICS:
            return "SKIP"

        written_regs = [
            r for r, _ in inst.getWrittenRegisters()
            if r.getName().lower() not in FLAG_REGISTERS
        ]
        written_mems = [m for m, _ in inst.getStoreAccess()]

        if not written_regs and not written_mems:
            all_tainted = True
            has_reads = False
            for r, _ in inst.getReadRegisters():
                if r.getName().lower() in FLAG_REGISTERS:
                    continue
                has_reads = True
                if not self.ctx.isRegisterTainted(r):
                    all_tainted = False
                    break
            for m, _ in inst.getLoadAccess():
                has_reads = True
                if not self.ctx.isMemoryTainted(m):
                    all_tainted = False
                    break
            return "JUNK" if has_reads and all_tainted else "CLEAN"

        total = len(written_regs) + len(written_mems)
        tainted = sum(1 for r in written_regs if self.ctx.isRegisterTainted(r))
        tainted += sum(1 for m in written_mems if self.ctx.isMemoryTainted(m))

        if tainted == 0:
            return "CLEAN"
        if tainted == total:
            return "SUSPECT" if mnem in CONTROL_FLOW_MNEMONICS else "JUNK"
        return "MIXED"

    # ------------------------------------------------------------------
    #  Call handling
    # ------------------------------------------------------------------

    def _handle_call(self):
        # Simulate CALL return: restore RSP (CALL pushed the return address)
        rsp = self.ctx.getConcreteRegisterValue(self.ctx.registers.rsp)
        self.ctx.setConcreteRegisterValue(self.ctx.registers.rsp, rsp + 8)

        # Untaint volatile registers (junk array taint does not survive calls)
        for name in VOLATILE_REGS:
            self.ctx.untaintRegister(self.ctx.getRegister(name))
        self.ctx.setConcreteRegisterValue(self.ctx.registers.rax, 0xC0FFEE)

        # Volatile registers also lose null status after a call
        for name in VOLATILE_REGS:
            self._null_regs.discard(name)

    # ------------------------------------------------------------------
    #  Run analysis
    # ------------------------------------------------------------------

    def run(self):
        """Analyze the function and highlight junk instructions."""
        self.junk_addrs.clear()
        self.null_addrs.clear()

        insn_list = []
        for block in self.func.basic_blocks:
            addr = block.start
            while addr < block.end:
                length = self.bv.get_instruction_length(addr)
                if length == 0:
                    break
                insn_list.append((addr, length))
                addr += length
        insn_list.sort(key=lambda x: x[0])

        for addr, length in insn_list:
            raw = bytes(self.bv.read(addr, length))
            inst = Instruction(addr, raw)

            try:
                self.ctx.processing(inst)
            except Exception:
                continue

            cls = self._classify(inst)

            if cls == "CALL":
                self._handle_call()
                self._update_null(inst)
                continue

            # Check null no-op BEFORE updating state (needs current state)
            is_null_nop = self._is_null_junk(inst)
            self._update_null(inst)

            if cls in ("JUNK", "MIXED"):
                self.junk_addrs.append((addr, length))
                self.func.set_auto_instr_highlight(
                    addr, HighlightStandardColor.RedHighlightColor
                )
            elif is_null_nop:
                self.null_addrs.append((addr, length))
                self.func.set_auto_instr_highlight(
                    addr, HighlightStandardColor.OrangeHighlightColor
                )

        total = len(insn_list)
        log_info(
            f"[Deobf] {self.func.name}: "
            f"junk_array={len(self.junk_addrs)} null_nop={len(self.null_addrs)} "
            f"/ {total} "
            f"({(len(self.junk_addrs)+len(self.null_addrs))/total*100:.0f}%)"
        )

    # ------------------------------------------------------------------
    #  Clear highlights
    # ------------------------------------------------------------------

    def clear(self):
        for addr, _ in self.junk_addrs + self.null_addrs:
            self.func.set_auto_instr_highlight(
                addr, HighlightStandardColor.NoHighlightColor
            )
        self.junk_addrs.clear()
        self.null_addrs.clear()

    # ------------------------------------------------------------------
    #  NOP-patch all identified junk
    # ------------------------------------------------------------------

    def patch(self):
        """
        NOP-patch all highlighted instructions (red + orange).
        Call run() first to identify junk instructions.
        """
        all_targets = self.junk_addrs + self.null_addrs
        if not all_targets:
            log_info(f"[Deobf] {self.func.name}: nothing to patch")
            return

        patched = 0
        for addr, size in all_targets:
            self.bv.write(addr, b'\x90' * size)
            patched += 1

        self.func.reanalyze()
        log_info(f"[Deobf] {self.func.name}: patched {patched} instructions to NOP")


# ======================================================================
# Main execution (Binary Ninja Script Console)
# ======================================================================

bv.begin_undo_actions()

d = JunkArrayDeobfuscator(bv, current_function)
d.run()
d.patch()

bv.update_analysis_and_wait()
bv.commit_undo_actions()
