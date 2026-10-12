"""
Stage 3 — Control Flow Flattening (CFF) Deobfuscator (Binary Ninja Plugin)

Recovers the original control flow graph from functions obfuscated with
control flow flattening.  CFF replaces direct branches between basic
blocks with a dispatcher loop: each "Original Basic Block" (OBB) updates
a state variable and jumps back to a central dispatcher, which routes
execution to the next OBB based on the state value.

Recovery algorithm:
  1. Identify the pre-dispatcher and dispatcher blocks by finding the
     basic block with the most incoming edges (pre-dispatcher) whose
     sole successor has exactly 2 incoming edges and multiple outgoing
     edges (dispatcher).
  2. Identify the state variable by tracing backward from the dispatcher
     to find a stack-variable store of a constant value.
  3. For each reference to the state variable, classify the OBB:
     - Unconditional: stores a constant next-state value.
     - Conditional: uses a CMOV instruction to select between two
       next-state values based on a flags condition.
  4. Build the recovered CFG by resolving state values to OBB addresses.
  5. Patch the binary: replace state-update + dispatcher-jump sequences
     with direct JMP/JCC instructions to the resolved target OBBs.

Environment: Binary Ninja Script Console.
"""

import binaryninja as bn
from binaryninja import *
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class OriginalBasicBlock:
    """
    Represents one Original Basic Block (OBB) in the recovered CFG.
    Each OBB corresponds to a real basic block that existed before
    the CFF transformation.
    """
    block: BasicBlock
    start_addr: int
    end_addr: int

    state_value: int

    is_conditional: bool = False
    condition_type: Optional[str] = None

    # State transitions (next OBB to execute)
    next_state_true: Optional[int] = None
    next_state_false: Optional[int] = None

    # Patching metadata
    state_change_addr: int = None
    state_change_size: int = 0
    cmov_instruction: Optional[str] = None
    cmov_operands: List[str] = field(default_factory=list)
    flags_setter_addr: Optional[int] = None
    patch_start_addr: Optional[int] = None
    patch_available_size: int = 0

    is_exit_block: bool = False

    decide_block: Optional[BasicBlock] = None
    decide_value: Optional[int] = None

    real_successors: List['OriginalBasicBlock'] = field(default_factory=list)

    def __hash__(self):
        return hash(self.start_addr)

    def __eq__(self, other):
        if not isinstance(other, OriginalBasicBlock):
            return False
        return self.start_addr == other.start_addr

    def __repr__(self):
        cond = f"{self.condition_type}" if self.is_conditional else "unconditional"
        t = f"0x{self.next_state_true:x}" if self.next_state_true else "None"
        f = f"0x{self.next_state_false:x}" if self.next_state_false else "None"
        return f"OBB(0x{self.start_addr:x}, state=0x{self.state_value:x}, {cond}, true={t}, false={f})"

    def is_ready_to_patch(self) -> bool:
        if self.is_exit_block:
            return True
        if not self.is_conditional:
            return self.next_state_true is not None
        return (self.next_state_true is not None
                and self.next_state_false is not None
                and self.condition_type is not None)

    def get_patch_info(self) -> Dict:
        return {
            'start_addr': self.start_addr,
            'patch_addr': self.patch_start_addr or self.state_change_addr,
            'patch_size': self.patch_available_size or self.state_change_size,
            'is_conditional': self.is_conditional,
            'condition': self.condition_type,
            'target_true': self.next_state_true,
            'target_false': self.next_state_false,
            'cmov_inst': self.cmov_instruction,
            'flags_addr': self.flags_setter_addr,
        }


class CFFDeobfuscator:
    """Main class for analyzing and deobfuscating CFF-protected functions."""

    # Mapping from CMOV mnemonics to their equivalent JCC mnemonics
    CMOV_TO_JCC = {
        'cmove': 'je', 'cmovz': 'je',
        'cmovne': 'jne', 'cmovnz': 'jne',
        'cmovl': 'jl', 'cmovnge': 'jl',
        'cmovle': 'jle', 'cmovng': 'jle',
        'cmovg': 'jg', 'cmovnle': 'jg',
        'cmovge': 'jge', 'cmovnl': 'jge',
        'cmova': 'ja', 'cmovnbe': 'ja',
        'cmovae': 'jae', 'cmovnb': 'jae',
        'cmovb': 'jb', 'cmovnae': 'jb',
        'cmovbe': 'jbe', 'cmovna': 'jbe',
        'cmovs': 'js', 'cmovns': 'jns',
        'cmovo': 'jo', 'cmovno': 'jno',
        'cmovp': 'jp', 'cmovnp': 'jnp',
    }

    # Near-JCC opcodes (0F 8x): second byte of the two-byte opcode
    JCC_OPCODES = {
        'jo': 0x80, 'jno': 0x81,
        'jb': 0x82, 'jnb': 0x83, 'jc': 0x82, 'jnc': 0x83,
        'jz': 0x84, 'jnz': 0x85, 'je': 0x84, 'jne': 0x85,
        'jbe': 0x86, 'ja': 0x87,
        'js': 0x88, 'jns': 0x89,
        'jp': 0x8A, 'jnp': 0x8B,
        'jl': 0x8C, 'jge': 0x8D,
        'jle': 0x8E, 'jg': 0x8F,
        'jae': 0x83, 'jnae': 0x82,
        'jnbe': 0x87, 'jna': 0x86,
        'jng': 0x8E, 'jnle': 0x8F,
        'jnl': 0x8D, 'jnge': 0x8C,
    }

    def __init__(self, bv: BinaryView, function: Function):
        self.bv = bv
        self.function = function
        self.predispatcher_block: Optional[BasicBlock] = None
        self.dispatcher_block: Optional[BasicBlock] = None
        self.state_variable: Optional[Variable] = None
        self.initial_state_value: Optional[int] = None

        self.obbs: Dict[int, OriginalBasicBlock] = {}       # state_value -> OBB
        self.obb_by_addr: Dict[int, OriginalBasicBlock] = {} # start_addr -> OBB

    # ==================================================================
    # Analysis entry point
    # ==================================================================

    def analyze(self) -> bool:
        """
        Analyze the function to identify CFF components.
        Returns True if a valid CFF pattern was found.
        """
        print(f"[*] Analyzing function: {self.function.name}")

        if not self._find_dispatcher_block():
            print("\t[!] Dispatcher block not found")
            return False

        if not self._identify_state_variable():
            print("\t[!] State variable not identified")
            return False

        if not self._analyze_obbs():
            print("\t[!] OBB analysis failed")
            return False

        return True

    # ==================================================================
    # Step 1: Find dispatcher block
    # ==================================================================

    def _find_dispatcher_block(self) -> bool:
        """
        Locate the pre-dispatcher and dispatcher blocks.

        The pre-dispatcher is the block with the most incoming edges
        (all OBBs jump back to it after updating state).  It has exactly
        one outgoing edge leading to the dispatcher, which uses the state
        variable to route execution.
        """
        try:
            candidates = [
                (len(b.incoming_edges), b)
                for b in self.function.basic_blocks
                if len(b.incoming_edges) >= 3
            ]
            candidates.sort(key=lambda x: x[0], reverse=True)

            for edge_count, predispatcher in candidates:
                if len(predispatcher.outgoing_edges) != 1:
                    continue

                dispatcher = predispatcher.outgoing_edges[0].target
                if len(dispatcher.incoming_edges) != 2:
                    continue
                if len(dispatcher.outgoing_edges) < 2:
                    continue

                self.predispatcher_block = predispatcher
                self.dispatcher_block = dispatcher
                return True

            print("\t[!] No valid pre-dispatcher/dispatcher pair found")
            return False
        except Exception as e:
            print(f"\t[!] Error finding dispatcher: {e}")
            return False

    # ==================================================================
    # Step 2: Identify the state variable
    # ==================================================================

    def _identify_state_variable(self) -> bool:
        try:
            if not self.dispatcher_block:
                return False

            var = self._find_state_var_from_disassembly()
            if var:
                self.state_variable = var
                return True
            return False
        except Exception as e:
            print(f"\t[!] Error identifying state variable: {e}")
            return False

    def _find_state_var_from_disassembly(self) -> Optional[Variable]:
        """
        Walk backward from the dispatcher block to find a STORE of a
        constant value to a stack variable — this is the initial state
        assignment that reveals the state variable identity.
        """
        try:
            if not self.dispatcher_block:
                return None

            current_addr = self.dispatcher_block.start

            for _ in range(20):
                llil = self.function.get_llil_at(current_addr)
                if llil and llil.operation == LowLevelILOperation.LLIL_STORE:
                    src = llil.src
                    if src.operation == LowLevelILOperation.LLIL_CONST:
                        disasm = self.bv.get_disassembly(current_addr)
                        var = int(disasm.split()[-1], 16)
                        if var:
                            self.initial_state_value = var & 0xFFFFFFFF
                            for tok in llil.tokens:
                                if tok.type == InstructionTextTokenType.StackVariableToken:
                                    return self.function.get_variable_by_name(tok.text)

                prev = self.function.get_instruction_containing_address(current_addr - 1)
                if prev == current_addr:
                    break
                current_addr = prev

            return None
        except Exception as e:
            print(f"\t[!] Error in state variable search: {e}")
            return None

    # ==================================================================
    # Step 3: Analyze all OBBs
    # ==================================================================

    def _analyze_obbs(self):
        """Identify and classify all Original Basic Blocks."""
        try:
            # Create entry OBB
            entry_block = self.function.basic_blocks[0]
            entry_obb = OriginalBasicBlock(
                block=entry_block,
                start_addr=entry_block.start,
                end_addr=entry_block.end,
                state_value=0,
                is_conditional=False,
                next_state_true=self.initial_state_value,
                state_change_addr=self.dispatcher_block.start,
            )
            self.obbs[0] = entry_obb
            self.obb_by_addr[entry_block.start] = entry_obb

            # Create exit OBBs (blocks with no outgoing edges)
            for exit_block in self._find_exit_blocks():
                decide_block = exit_block.incoming_edges[0].source if exit_block.incoming_edges else None
                decide_value = self._find_decide_value(decide_block) if decide_block else None

                obb = OriginalBasicBlock(
                    block=exit_block,
                    start_addr=exit_block.start,
                    end_addr=exit_block.end,
                    state_value=decide_value,
                    is_exit_block=True,
                    decide_block=decide_block,
                    decide_value=decide_value,
                )
                if decide_value:
                    self.obbs[decide_value] = obb
                self.obb_by_addr[exit_block.start] = obb

            # Analyze all references to the state variable
            references = self.function.get_mlil_var_refs(self.state_variable)
            if len(references) == 0:
                # Binary Ninja may suffix the variable name with _1
                self.state_variable = self.function.get_variable_by_name(
                    self.state_variable.name + '_1'
                )
                references = self.function.get_mlil_var_refs(self.state_variable)

            for ref in references:
                ref_address = ref.address
                ref_block = self.function.get_basic_block_at(ref_address)

                if ref_block == self.dispatcher_block:
                    continue
                if ref_block == self.function.basic_blocks[0]:
                    continue

                decide_block = self._find_decide_block(ref_block)
                decide_value = self._find_decide_value(decide_block) if decide_block else None

                llil = self.function.get_llil_at(ref_address)
                if llil.operation != LowLevelILOperation.LLIL_STORE:
                    continue

                # Unconditional OBB: state = constant
                if llil.src.operation == LowLevelILOperation.LLIL_CONST:
                    next_state = llil.src.constant & 0xFFFFFFFF
                    obb = OriginalBasicBlock(
                        block=ref_block,
                        start_addr=ref_block.start,
                        end_addr=ref_block.end,
                        state_value=decide_value,
                        is_conditional=False,
                        next_state_true=next_state,
                        state_change_addr=ref_address,
                        decide_block=decide_block,
                        decide_value=decide_value,
                    )
                    if decide_value:
                        self.obbs[decide_value] = obb
                    self.obb_by_addr[ref_block.start] = obb

                # Conditional OBB: state = register (from CMOV)
                elif llil.src.operation == LowLevelILOperation.LLIL_REG:
                    cmov_info = self._analyze_cmov(ref_block, ref_address)
                    reg_name = llil.src.tokens[0].text

                    try:
                        possible_values = list(llil.get_possible_reg_values(reg_name).values)
                    except Exception:
                        cmov_addr = cmov_info['patch_address'] if cmov_info else None
                        if cmov_addr:
                            states = self._get_conditional_states_by_slicing(ref_block, cmov_addr)
                            if states:
                                true_val, false_val = states
                                possible_values = [true_val, false_val]
                                cmov_info['true_value'] = true_val
                                cmov_info['false_value'] = false_val
                            else:
                                return False
                        else:
                            return False

                    if len(possible_values) >= 2:
                        obb = OriginalBasicBlock(
                            block=ref_block,
                            start_addr=ref_block.start,
                            end_addr=ref_block.end,
                            state_value=decide_value,
                            is_conditional=True,
                            condition_type=cmov_info['type'] if cmov_info else None,
                            next_state_true=(cmov_info['true_value'] & 0xFFFFFFFF
                                             if cmov_info else possible_values[0] & 0xFFFFFFFF),
                            next_state_false=(cmov_info['false_value'] & 0xFFFFFFFF
                                              if cmov_info else possible_values[1] & 0xFFFFFFFF),
                            cmov_instruction=cmov_info['instruction'] if cmov_info else None,
                            cmov_operands=cmov_info['operands'] if cmov_info else [],
                            state_change_addr=ref_address,
                            decide_block=decide_block,
                            decide_value=decide_value,
                            patch_start_addr=cmov_info['patch_address'] if cmov_info else None,
                        )
                        if decide_value:
                            self.obbs[decide_value] = obb
                        self.obb_by_addr[ref_block.start] = obb

            return True
        except Exception as e:
            print(f"\t[!] Error during OBB analysis: {e}")
            return False

    # ==================================================================
    # Helper: find the "decide block" (contains `sub eax, <state_value>`)
    # ==================================================================

    def _find_decide_block(self, target_block: BasicBlock) -> Optional[BasicBlock]:
        try:
            current = target_block
            for _ in range(10):
                parent = current.incoming_edges[0].source
                if self._has_decide_pattern(parent):
                    return parent
                current = parent
            return None
        except Exception:
            return None

    def _has_decide_pattern(self, block: BasicBlock) -> bool:
        """Check if the block contains a MOV + SUB pattern (dispatcher case entry)."""
        try:
            lines = block.get_disassembly_text()
            first = str(lines[0]).lower()
            second = str(lines[1]).lower() if len(lines) > 1 else ""
            third = str(lines[2]).lower() if len(lines) > 2 else ""

            if 'mov' in first and 'sub' in second:
                return True
            if 'mov' in first and 'mov' in second and 'sub' in third:
                return True
            return False
        except Exception:
            return False

    def _find_decide_value(self, block: BasicBlock) -> Optional[int]:
        """Extract the state value from a `sub eax, <value>` instruction."""
        try:
            for text in block.disassembly_text:
                line = str(text)
                if "sub" in line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        val_str = parts[1].strip()
                        return int(val_str, 16) if val_str.startswith('0x') else int(val_str)
            return None
        except Exception:
            return None

    def _find_exit_blocks(self) -> List[BasicBlock]:
        return [b for b in self.function.basic_blocks if len(b.outgoing_edges) == 0]

    # ==================================================================
    # CMOV analysis
    # ==================================================================

    def _analyze_cmov(self, block: BasicBlock, store_addr: int) -> Optional[Dict]:
        """
        Walk backward from a state-variable store to find the preceding
        CMOV instruction and extract the condition type and operand values.
        """
        try:
            current_addr = store_addr
            cmov_types = [
                'cmove', 'cmovne', 'cmovl', 'cmovle', 'cmovg', 'cmovge',
                'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmovs', 'cmovns',
            ]

            for _ in range(10):
                prev = self.function.get_instruction_containing_address(current_addr - 1)
                if prev == current_addr:
                    break
                current_addr = prev

                line = self.bv.get_disassembly(current_addr).lower()
                for cmov in cmov_types:
                    if cmov in line:
                        parts = self.bv.get_disassembly(current_addr).split()
                        if len(parts) >= 3:
                            cond_type = cmov[4:]  # e.g., 'cmovne' -> 'ne'
                            operands = [op.strip().rstrip(',') for op in ' '.join(parts[1:]).split(',')]
                            operands = list(reversed(operands))
                            llil = self.function.get_llil_at(current_addr)
                            return {
                                'type': cond_type,
                                'instruction': cmov,
                                'operands': operands,
                                'true_value': llil.get_reg_value(operands[0]).value,
                                'false_value': llil.get_reg_value(operands[1]).value,
                                'patch_address': current_addr,
                            }
            return None
        except Exception as e:
            print(f"\t[!] Error analyzing CMOV: {e}")
            return None

    def _get_conditional_states_by_slicing(self, block, cmov_addr) -> Optional[Tuple[int, int]]:
        """Fallback: use backward slicing to resolve CMOV operand values."""
        try:
            disasm = self.bv.get_disassembly(cmov_addr)
            parts = disasm.split()
            if len(parts) < 3:
                return None

            operands = [op.strip().rstrip(',') for op in ' '.join(parts[1:]).split(',')]
            if len(operands) != 2:
                return None

            dest_reg, src_reg = operands[0], operands[1]
            false_value = self._trace_reg_to_constant(block, cmov_addr, dest_reg)
            true_value = self._trace_reg_to_constant(block, cmov_addr, src_reg)

            if false_value is not None and true_value is not None:
                return (true_value, false_value)
            return None
        except Exception:
            return None

    def _trace_reg_to_constant(self, block, start_addr, register) -> Optional[int]:
        """Trace a register backward to find the constant value assigned to it."""
        try:
            current_addr = start_addr
            target_reg = register.lower()

            for _ in range(100):
                prev = self.function.get_instruction_containing_address(current_addr - 1)
                if prev == current_addr or prev < block.start:
                    break
                current_addr = prev

                llil = self.function.get_llil_at(current_addr)
                if not llil:
                    continue

                if llil.operation == LowLevelILOperation.LLIL_SET_REG:
                    if self._is_same_register(str(llil.dest), target_reg):
                        src = llil.src
                        if src.operation == LowLevelILOperation.LLIL_LOAD:
                            offset = self._get_stack_offset(src.src)
                            if offset is not None:
                                return self._find_last_store_to_stack(block, current_addr, offset)
            return None
        except Exception:
            return None

    @staticmethod
    def _is_same_register(reg1: str, reg2: str) -> bool:
        """Check if two register names refer to the same physical register."""
        reg_map = {}
        for parent, children in {
            'rax': ['rax', 'eax', 'ax', 'al', 'ah'],
            'rbx': ['rbx', 'ebx', 'bx', 'bl', 'bh'],
            'rcx': ['rcx', 'ecx', 'cx', 'cl', 'ch'],
            'rdx': ['rdx', 'edx', 'dx', 'dl', 'dh'],
            'rsi': ['rsi', 'esi', 'si', 'sil'],
            'rdi': ['rdi', 'edi', 'di', 'dil'],
            'rbp': ['rbp', 'ebp', 'bp', 'bpl'],
            'rsp': ['rsp', 'esp', 'sp', 'spl'],
            **{f'r{n}': [f'r{n}', f'r{n}d', f'r{n}w', f'r{n}b'] for n in range(8, 16)},
        }.items():
            for c in children:
                reg_map[c] = parent

        return reg_map.get(reg1.lower().strip(), reg1) == reg_map.get(reg2.lower().strip(), reg2)

    @staticmethod
    def _get_stack_offset(addr_expr) -> Optional[int]:
        if not addr_expr:
            return None
        if addr_expr.operation == LowLevelILOperation.LLIL_SUB:
            if len(addr_expr.operands) >= 2:
                off = addr_expr.operands[1]
                if off.operation == LowLevelILOperation.LLIL_CONST:
                    return -off.constant
        elif addr_expr.operation == LowLevelILOperation.LLIL_ADD:
            if len(addr_expr.operands) >= 2:
                off = addr_expr.operands[1]
                if off.operation == LowLevelILOperation.LLIL_CONST:
                    return off.constant
        return None

    def _find_last_store_to_stack(self, block, before_addr, stack_offset) -> Optional[int]:
        """Find the most recent store of a constant to the given stack offset."""
        try:
            current_addr = before_addr
            for _ in range(500):
                prev = self.function.get_instruction_containing_address(current_addr - 1)
                if prev == current_addr:
                    break
                current_addr = prev

                llil = self.function.get_llil_at(current_addr)
                if not llil or llil.operation != LowLevelILOperation.LLIL_STORE:
                    continue
                if llil.dest.operation == LowLevelILOperation.LLIL_REG:
                    continue
                if self._get_stack_offset(llil.dest) == stack_offset:
                    src = llil.src
                    if src.operation == LowLevelILOperation.LLIL_CONST:
                        return src.constant & 0xFFFFFFFF
                    elif src.operation == LowLevelILOperation.LLIL_REG:
                        return llil.get_reg_value(llil.src.src).value
            return None
        except Exception:
            return None

    # ==================================================================
    # Build recovered CFG
    # ==================================================================

    def build_real_cfg(self):
        """Resolve state-value transitions into direct OBB -> OBB edges."""
        try:
            for state_value, obb in self.obbs.items():
                if obb.is_exit_block:
                    continue
                if not obb.is_conditional:
                    if obb.next_state_true and obb.next_state_true in self.obbs:
                        obb.real_successors.append(self.obbs[obb.next_state_true])
                else:
                    if obb.next_state_true and obb.next_state_true in self.obbs:
                        obb.real_successors.append(self.obbs[obb.next_state_true])
                    if obb.next_state_false and obb.next_state_false in self.obbs:
                        obb.real_successors.append(self.obbs[obb.next_state_false])
        except Exception as e:
            print(f"\t[!] Error building CFG: {e}")

    # ==================================================================
    # Binary patching
    # ==================================================================

    def patch_binary(self):
        """Patch the binary to restore direct control flow between OBBs."""
        print("[*] Patching binary...")
        try:
            patched = 0

            # Patch entry block first
            if 0 in self.obbs:
                if self._patch_entry_block(self.obbs[0]):
                    patched += 1

            for state_value, obb in sorted(self.obbs.items()):
                if state_value == 0 or obb.is_exit_block:
                    continue
                if not obb.is_ready_to_patch():
                    continue
                if obb.is_conditional:
                    if self._patch_conditional_block(obb):
                        patched += 1
                else:
                    if self._patch_unconditional_block(obb):
                        patched += 1
        except Exception as e:
            print(f"\t[!] Error during patching: {e}")

    def _patch_entry_block(self, entry_obb) -> bool:
        """Replace `mov [state_var], initial_value` with `jmp <first_OBB>`."""
        try:
            if not entry_obb.next_state_true or entry_obb.next_state_true not in self.obbs:
                return False

            first_block = self.obbs[entry_obb.next_state_true]
            patch_addr = entry_obb.state_change_addr

            jmp_target = first_block.start_addr
            jmp_offset = jmp_target - (patch_addr + 5)
            jmp_bytes = bytes([0xE9]) + jmp_offset.to_bytes(4, byteorder='little', signed=True)
            self.bv.write(patch_addr, jmp_bytes)
            return True
        except Exception as e:
            print(f"\t[!] Error patching entry block: {e}")
            return False

    def _patch_conditional_block(self, obb) -> bool:
        """
        Replace CMOV + state store + JMP dispatcher with:
          JCC rel32 <true_target>   (6 bytes)
          JMP rel32 <false_target>  (5 bytes)
        """
        try:
            if obb.next_state_true not in self.obbs or obb.next_state_false not in self.obbs:
                return False

            target_true = self.obbs[obb.next_state_true]
            target_false = self.obbs[obb.next_state_false]

            patch_addr = obb.patch_start_addr
            if not patch_addr:
                return False

            jcc_mnemonic = self.CMOV_TO_JCC.get(obb.cmov_instruction)
            if not jcc_mnemonic:
                return False

            pos = patch_addr

            # JCC rel32 (6 bytes)
            jcc_opcode = self.JCC_OPCODES.get(jcc_mnemonic, 0x85)
            jcc_offset = target_true.start_addr - (pos + 6)
            self.bv.write(pos, bytes([0x0F, jcc_opcode]) +
                          jcc_offset.to_bytes(4, byteorder='little', signed=True))
            pos += 6

            # JMP rel32 (5 bytes)
            jmp_offset = target_false.start_addr - (pos + 5)
            self.bv.write(pos, bytes([0xE9]) +
                          jmp_offset.to_bytes(4, byteorder='little', signed=True))

            return True
        except Exception as e:
            print(f"\t[!] Error patching conditional block: {e}")
            return False

    def _patch_unconditional_block(self, obb) -> bool:
        """Replace `mov [state_var], value; jmp dispatcher` with `jmp <target>; NOP...`."""
        try:
            if obb.next_state_true not in self.obbs:
                return False

            target = self.obbs[obb.next_state_true]
            patch_addr = obb.state_change_addr
            if not patch_addr:
                return False

            available = obb.block.end - patch_addr
            if available < 5:
                return False

            jmp_offset = target.start_addr - (patch_addr + 5)
            jmp_bytes = bytes([0xE9]) + jmp_offset.to_bytes(4, byteorder='little', signed=True)
            self.bv.write(patch_addr, jmp_bytes)

            remaining = available - 5
            if remaining > 0:
                self.bv.write(patch_addr + 5, b'\x90' * remaining)

            return True
        except Exception as e:
            print(f"\t[!] Error patching unconditional block: {e}")
            return False


# ======================================================================
# Public API
# ======================================================================

def deobfuscate_cff(bv: BinaryView, function: Function) -> bool:
    """
    Deobfuscate a single CFF-protected function.
    Returns True on success.
    """
    if not function:
        print(f"[!] Invalid function")
        return False

    deobf = CFFDeobfuscator(bv, function)

    if deobf.analyze():
        print(f"[+] Analysis succeeded: {function.name}")
        deobf.build_real_cfg()
        deobf.patch_binary()
        return True
    else:
        print(f"\t[!] Analysis failed: {function.name}")
        return False


# ======================================================================
# Main execution (Binary Ninja Script Console)
# ======================================================================

if __name__ == "__main__":
    bv.begin_undo_actions()

    total = len(bv.functions)
    success = 0
    print(f"[*] Deobfuscating all functions ({total} total)...\n")

    for func in bv.functions:
        if deobfuscate_cff(bv, func):
            print(f"[+] Deobfuscated: {func.name}\n")
            success += 1
        else:
            print(f"[-] Skipped: {func.name}\n")

    print(f"\n[+] Deobfuscated {success}/{total} functions.")

    bv.update_analysis_and_wait()
    bv.commit_undo_actions()
