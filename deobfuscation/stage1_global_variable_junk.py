"""
Stage 1 — Global Variable Junk Code Remover (Binary Ninja Plugin)

Identifies and patches junk code in the Stage 1 loader that references
heavily-used global variables. The implant's obfuscator inserts spurious
reads/writes to designated global variables (selected by having >= 100
cross-references) to bloat the binary and hinder static analysis.

Algorithm:
  1. Enumerate all data variables with >= 100 code references.
  2. For each reference, perform a backward register-dependency trace
     (using the Triton symbolic execution engine) to identify the full
     set of instructions that exist solely to compute or consume the
     junk global variable.
  3. NOP-patch every identified junk instruction.
     - Plain instructions are replaced with 0x90 (NOP).
     - CALL instructions are replaced with `xor rax, rax` + NOP padding
       to preserve stack alignment and neutralize the return value.

Environment: Binary Ninja Script Console (requires the `triton` package).
"""

import copy
import json
from triton import TritonContext, ARCH, Instruction


class BackwardTrace:
    """
    Triton-based backward register-dependency tracer.

    Starting from a given instruction address, walks backward through
    the containing function to identify every instruction that
    contributes to the computation consumed at the start address.

    The trace tracks a "relevant register set" (rel): any instruction
    that writes to a register in this set is marked as junk, and its
    read operands are added to the set.  The trace terminates when the
    set is empty or the instruction limit (100) is reached.
    """

    INITIAL_RSP = 0x100000
    FLAGS = ('cf', 'pf', 'af', 'zf', 'sf', 'df', 'of')

    def __init__(self):
        self.tx = TritonContext()
        self.tx.setArchitecture(ARCH.X86_64)
        self.tx.setMode(True, True)

        self.instructions = [{'ins': None, 'state': {'regs': {}, 'mem': {}, 'rsp': self.INITIAL_RSP}}]
        self.prev_instr_cache = {}

    def setup(self):
        self.tx.setConcreteRegisterValue(self.tx.registers.rsp, self.INITIAL_RSP)

    # ------------------------------------------------------------------
    # Address helpers
    # ------------------------------------------------------------------

    def _build_prev_instr_map(self, addr):
        """Build and cache a mapping of address -> previous instruction address within the function."""
        func = bv.get_functions_containing(addr)[0]
        if not func:
            return None
        if func in self.prev_instr_cache:
            return self.prev_instr_cache[func]

        prev_map = {}
        prev_addr = None
        cur = func.address_ranges[0].start
        end = func.address_ranges[0].end

        while cur < end:
            instr_len = bv.get_instruction_length(cur)
            if prev_addr is not None:
                prev_map[cur] = prev_addr
            prev_addr = cur
            cur += instr_len

        self.prev_instr_cache[func] = prev_map
        return prev_map

    def get_previous_instruction_address(self, addr):
        prev_map = self._build_prev_instr_map(addr)
        if prev_map is None:
            return None
        return prev_map.get(addr, None)

    # ------------------------------------------------------------------
    # Highlighting
    # ------------------------------------------------------------------

    def set_highlight(self, address):
        func = bv.get_functions_containing(address)[0]
        func.set_user_instr_highlight(address, HighlightStandardColor.MagentaHighlightColor)

    def clear_highlight(self, address):
        func = bv.get_functions_containing(address)[0]
        func.set_user_instr_highlight(address, HighlightStandardColor.NoHighlightColor)

    # ------------------------------------------------------------------
    # Triton instruction side-effect helpers
    # ------------------------------------------------------------------

    def _get_written_regs(self, ins):
        regs = set()
        for reg, ast in ins.getWrittenRegisters():
            if reg.getName() not in ("rip", "rsp"):
                regs.add(self.tx.getParentRegister(reg))
        return regs

    def _get_read_regs(self, ins):
        regs = set()
        for reg, ast in ins.getReadRegisters():
            if reg.getName() not in ("rip", "rsp"):
                if ins.getDisassembly().startswith("ro") and reg.getName() in ("of", "cf"):
                    continue
                regs.add(self.tx.getParentRegister(reg))
        return regs

    def _get_store_accesses(self, ins):
        for mem, ast in ins.getStoreAccess():
            yield mem

    # ------------------------------------------------------------------
    # Instruction state tracking
    # ------------------------------------------------------------------

    def _record_instruction(self, ins, prev_rsp):
        entry = {
            'ins': ins,
            'state': copy.deepcopy(self.instructions[-1]['state']),
        }
        entry['state']['rsp'] = prev_rsp

        for reg in self._get_written_regs(ins):
            entry['state']['regs'][reg.getName()] = len(self.instructions)
        for mem in self._get_store_accesses(ins):
            entry['state']['mem'][mem.getAddress()] = len(self.instructions)

        # Evict stack slots that have gone out of scope (RSP moved above them)
        cur_rsp = self.tx.getConcreteRegisterValue(self.tx.registers.rsp)
        for addr in [a for a in entry['state']['mem'] if cur_rsp > a]:
            del entry['state']['mem'][addr]

        self.instructions.append(entry)
        return len(self.instructions) - 1

    # ------------------------------------------------------------------
    # Register normalization (sub-register -> 64-bit canonical name)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_register(reg):
        reg = reg.lower()
        if reg.startswith('e') and len(reg) == 3:
            return 'r' + reg[1:]
        elif reg.endswith('d'):
            return reg[:-1]
        elif reg.endswith('h') or reg.endswith('l'):
            return reg[:-1]
        return reg

    def _reg_in_list(self, register, target_list):
        norm = self._normalize_register(register)
        return any(self._normalize_register(r) == norm for r in target_list)

    def _remove_reg_from_list(self, register, target_list):
        norm = self._normalize_register(register)
        target_list[:] = [r for r in target_list if self._normalize_register(r) != norm]
        return target_list

    # ------------------------------------------------------------------
    # Core backward trace
    # ------------------------------------------------------------------

    def backward_trace(self, trace_address, argc=0):
        """
        Trace backward from `trace_address` and return all addresses
        that are part of the dependency chain.

        Args:
            trace_address: Starting instruction address.
            argc: If 1, also track RCX (first argument in x64 Windows ABI)
                  as an additional dependency root.

        Returns:
            List of instruction addresses identified as junk.
        """
        self.setup()
        argv = ['rcx'] if argc == 1 else []
        rel = []
        result = []

        # Process the starting instruction
        op = bv.read(trace_address, bv.get_instruction_length(trace_address))
        ins = Instruction()
        ins.setOpcode(op)
        ins.setAddress(trace_address)
        self.tx.processing(ins)

        result.append(trace_address)
        self.set_highlight(trace_address)

        # Seed the relevant-register set with the start instruction's reads
        for reg in self._get_read_regs(ins):
            name = reg.getName()
            if name not in self.FLAGS and not self._reg_in_list(name, rel):
                rel.append(name)

        prev_rsp = self.tx.getConcreteRegisterValue(self.tx.registers.rsp)
        self._record_instruction(ins, prev_rsp)
        trace_address = self.get_previous_instruction_address(trace_address)

        while trace_address is not None:
            if len(result) > 100:
                break

            op = bv.read(trace_address, bv.get_instruction_length(trace_address))
            ins = Instruction()
            ins.setOpcode(op)
            ins.setAddress(trace_address)
            prev_rsp = self.tx.getConcreteRegisterValue(self.tx.registers.rsp)
            self.tx.processing(ins)

            read_names = [x.getName() for x in self._get_read_regs(ins)]
            write_names = [x.getName() for x in self._get_written_regs(ins)]

            # Mark instruction if it writes to any tracked register
            marked = any(
                w not in self.FLAGS and (self._reg_in_list(w, rel) or self._reg_in_list(w, argv))
                for w in write_names
            )

            if marked:
                result.append(trace_address)
                self.set_highlight(trace_address)

                for r in read_names:
                    if r not in self.FLAGS and not self._reg_in_list(r, rel):
                        rel.append(r)

                for w in write_names:
                    if w in self.FLAGS:
                        continue
                    if self._reg_in_list(w, argv):
                        self._remove_reg_from_list(w, argv)
                    if self._reg_in_list(w, rel):
                        is_also_read = self._reg_in_list(w, read_names)
                        # `xor reg, reg` zeroes the register; no data dependency on old value
                        if "xor" in ins.getDisassembly().lower():
                            parts = ins.getDisassembly().lower().split()
                            if len(parts) >= 3 and is_also_read:
                                if parts[1].replace(',', '') == parts[2]:
                                    is_also_read = False
                        if not is_also_read:
                            self._remove_reg_from_list(w, rel)

            self._record_instruction(ins, prev_rsp)

            if not rel and not argv:
                break

            trace_address = self.get_previous_instruction_address(trace_address)

        return result


# ======================================================================
# Identify candidate global variables (junk sinks)
# ======================================================================

def get_highly_referenced_data_vars(bv, min_refs=100):
    """
    Return data variables that have an unusually high number of code
    references — a hallmark of obfuscator-inserted junk globals.
    """
    results = {}
    for addr, data_var in bv.data_vars.items():
        refs = list(bv.get_code_refs(addr))
        if len(refs) < min_refs:
            continue
        results[addr] = {
            'type': str(data_var.type),
            'symbol': data_var.symbol.name if data_var.symbol else None,
            'references': [
                {
                    'address': ref.address,
                    'function': ref.function.name if ref.function else None,
                    'arch': ref.arch.name if ref.arch else None,
                }
                for ref in refs
            ],
        }
    return results


# ======================================================================
# Main execution (Binary Ninja Script Console)
# ======================================================================

bv.begin_undo_actions()
patch = []

results = get_highly_referenced_data_vars(bv)
for addr, info in results.items():
    print(f"Variable 0x{addr:x}: {info['symbol']} - {len(info['references'])} refs")

    if info['symbol'] is None:
        # Unnamed variable: trace each DWORD-accessing reference
        for ref in info['references']:
            if 'dword' == bv.get_disassembly(ref['address']).split()[1]:
                tracer = BackwardTrace()
                for a in tracer.backward_trace(ref['address']):
                    patch.append(a)
            else:
                patch.append(ref['address'])
    else:
        # Named variable (likely a function pointer sink): trace with argc=1
        for ref in info['references']:
            tracer = BackwardTrace()
            for a in tracer.backward_trace(ref['address'], argc=1):
                patch.append(a)

if patch:
    print(f"Total {len(patch)} addresses to patch...")
    for addr in patch:
        insn = bv.get_disassembly(addr)
        length = bv.get_instruction_length(addr)
        if 'call' in insn:
            # Neutralize CALL: `xor rax, rax` (3 bytes) + NOP padding
            patchbyte = b"\x48\x31\xC0" + b'\x90' * (length - 3)
        else:
            patchbyte = b'\x90' * length
        bv.write(addr, patchbyte)

bv.update_analysis_and_wait()
bv.commit_undo_actions()
