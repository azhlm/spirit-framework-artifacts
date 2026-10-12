"""
Stage 2 — Junk Code Remover (Binary Ninja Plugin)

Identifies and NOP-patches junk code in the Stage 2 implant DLL.
Operates identically to the Stage 1 global-variable junk remover
(see stage1_global_variable_junk.py) but uses a lower reference-count
threshold (>= 20) suited to the Stage 2 binary's obfuscation density.

Algorithm:
  1. Enumerate all data variables with >= 20 code references.
  2. For each reference, backward-trace register dependencies via Triton
     to identify the full junk instruction chain.
  3. NOP-patch identified instructions (CALLs become `xor rax, rax` + NOPs).

Includes optional JSON export of patch addresses for offline tooling.

Environment: Binary Ninja Script Console (requires the `triton` package).
"""

import copy
import json
from triton import TritonContext, ARCH, Instruction


def save_patch_info(patches, json_path):
    """Export patch locations to JSON for use with external patch tools."""
    json_data = [
        {'address': hex(p['address']), 'length': p['length']}
        for p in patches
    ]
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Patch info saved to {json_path}")


def collect_and_save_patches(patch_addresses, json_path):
    """Collect instruction lengths and save as file-offset patch list."""
    patches = []
    for va in patch_addresses:
        length = bv.get_instruction_length(va)
        pa = bv.get_data_offset_for_address(va)
        patches.append({'address': pa, 'length': length})
    save_patch_info(patches, json_path)


class BackwardTrace:
    """
    Triton-based backward register-dependency tracer.

    Starting from a given instruction address, walks backward through
    the containing function to identify every instruction that
    contributes to the computation consumed at the start address.
    See stage1_global_variable_junk.py for detailed documentation.
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

    def _build_prev_instr_map(self, addr):
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

    def set_highlight(self, address):
        func = bv.get_functions_containing(address)[0]
        func.set_user_instr_highlight(address, HighlightStandardColor.MagentaHighlightColor)

    def clear_highlight(self, address):
        func = bv.get_functions_containing(address)[0]
        func.set_user_instr_highlight(address, HighlightStandardColor.NoHighlightColor)

    def _record_instruction(self, ins, prev_rsp):
        entry = {
            'ins': ins,
            'state': copy.deepcopy(self.instructions[-1]['state']),
        }
        entry['state']['rsp'] = prev_rsp

        for reg, ast in ins.getWrittenRegisters():
            if reg.getName() not in ("rip", "rsp"):
                par = self.tx.getParentRegister(reg)
                entry['state']['regs'][par.getName()] = len(self.instructions)

        for mem, ast in ins.getStoreAccess():
            entry['state']['mem'][mem.getAddress()] = len(self.instructions)

        cur_rsp = self.tx.getConcreteRegisterValue(self.tx.registers.rsp)
        for addr in [a for a in entry['state']['mem'] if cur_rsp > a]:
            del entry['state']['mem'][addr]

        self.instructions.append(entry)
        return len(self.instructions) - 1

    def _get_read_regs(self, ins):
        regs = set()
        for reg, ast in ins.getReadRegisters():
            if reg.getName() not in ("rip", "rsp"):
                if ins.getDisassembly().startswith("ro") and reg.getName() in ("of", "cf"):
                    continue
                regs.add(self.tx.getParentRegister(reg))
        return regs

    def _get_written_regs(self, ins):
        regs = set()
        for reg, ast in ins.getWrittenRegisters():
            if reg.getName() not in ("rip", "rsp"):
                regs.add(self.tx.getParentRegister(reg))
        return regs

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

    def backward_trace(self, trace_address, argc=0):
        """
        Trace backward from trace_address, returning all junk instruction addresses.

        Args:
            trace_address: Starting instruction address.
            argc: If 1, also track RCX as an additional dependency root.
        """
        self.setup()
        argv = ['rcx'] if argc == 1 else []
        rel = []
        result = []

        op_len = bv.get_instruction_length(trace_address)
        op = bv.read(trace_address, op_len)
        ins = Instruction()
        ins.setOpcode(op)
        ins.setAddress(trace_address)
        self.tx.processing(ins)

        result.append(trace_address)
        self.set_highlight(trace_address)

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

            op_len = bv.get_instruction_length(trace_address)
            op = bv.read(trace_address, op_len)
            ins = Instruction()
            ins.setOpcode(op)
            ins.setAddress(trace_address)
            prev_rsp = self.tx.getConcreteRegisterValue(self.tx.registers.rsp)
            self.tx.processing(ins)

            read_names = [x.getName() for x in self._get_read_regs(ins)]
            write_names = [x.getName() for x in self._get_written_regs(ins)]

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


def get_highly_referenced_data_vars(bv, min_refs=20):
    """Return data variables with >= min_refs code references."""
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
        for ref in info['references']:
            if 'dword' == bv.get_disassembly(ref['address']).split()[1]:
                rt = BackwardTrace()
                for a in rt.backward_trace(ref['address']):
                    patch.append(a)
            else:
                patch.append(ref['address'])
    else:
        for ref in info['references']:
            rt = BackwardTrace()
            for a in rt.backward_trace(ref['address'], argc=1):
                patch.append(a)

if patch:
    print(f"Total {len(patch)} addresses to patch...")
    for addr in patch:
        insn = bv.get_disassembly(addr)
        length = bv.get_instruction_length(addr)
        if 'call' in insn:
            patchbyte = b"\x48\x31\xC0" + b'\x90' * (length - 3)
        else:
            patchbyte = b'\x90' * length
        bv.write(addr, patchbyte)

bv.update_analysis_and_wait()
bv.commit_undo_actions()
