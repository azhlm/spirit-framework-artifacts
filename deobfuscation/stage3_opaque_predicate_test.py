"""
Stage 3 — Opaque Predicate Deobfuscator: Test & Debug Utilities

Interactive helpers for testing the opaque-predicate deobfuscation logic
on individual variable pairs before running the batch processor.  Useful
for validating the algorithm on new samples or investigating edge cases.

Usage (Binary Ninja Script Console):

    # Test a single pair of state variables
    test_single_pair(bv, 0xADDR1, 0xADDR2)

    # Test within a specific function
    test_single_pair(bv, 0xADDR1, 0xADDR2, "sub_...")

    # Preview patches without applying (dry run)
    test_and_apply_patch(bv, 0xADDR1, 0xADDR2, dry_run=True)

    # Apply patches for a specific pair
    test_and_apply_patch(bv, 0xADDR1, 0xADDR2, dry_run=False)

    # Find all candidate pairs referenced by a function
    pairs = find_pairs_in_function(bv, "sub_...")
"""

from capstone import Cs, CS_ARCH_X86, CS_MODE_64


def find_jne_target_from_ref(bv, ref_addr):
    """
    Locate the resolved JNE target from a code reference to a state
    variable.  See stage3_opaque_predicate.py for full documentation.
    """
    func = bv.get_functions_containing(ref_addr)[0]
    bb = func.get_basic_block_at(ref_addr)
    last_insn = str(bb.get_disassembly_text()[-1])

    if "jne" in last_insn:
        last_insn_addr = bb.get_instruction_containing_address(bb.end - 1)[1]
        current_addr = last_insn_addr
        previous_addr = current_addr

        keep_addr = None
        target_addr = int(last_insn.split()[-1], 16)

        while True:
            current_addr = bb.get_instruction_containing_address(current_addr - 1)[1]
            current_insn = str(bv.get_disassembly(current_addr))
            if "mov" not in current_insn:
                keep_addr = previous_addr
                break
            previous_addr = current_addr

        if bb.outgoing_edges and len(bb.outgoing_edges) == 1:
            next_bb = bb.outgoing_edges[0].target
            next_last_insn = str(next_bb.get_disassembly_text()[-1])
            if "jne" in next_last_insn:
                target_addr = int(next_last_insn.split()[-1], 16)

        if keep_addr and target_addr:
            if keep_addr == last_insn_addr:
                return [target_addr]
            return [keep_addr, last_insn_addr, target_addr]

    elif "jmp" in last_insn and bb.index == 0:
        last_insn_addr = bb.get_instruction_containing_address(bb.end - 1)[1]
        current_addr = last_insn_addr
        previous_addr = current_addr

        keep_addr = None
        target_addr = None

        while True:
            current_addr = bb.get_instruction_containing_address(current_addr - 1)[1]
            current_insn = str(bv.get_disassembly(current_addr))
            try:
                int(current_insn.split()[-1], 16)
                keep_addr = previous_addr
                break
            except ValueError:
                previous_addr = current_addr
                continue

        if bb.outgoing_edges and len(bb.outgoing_edges) == 1:
            next_bb = bb.outgoing_edges[0].target
            next_last_insn = str(next_bb.get_disassembly_text()[-1])
            if "jne" in next_last_insn:
                target_addr = int(next_last_insn.split()[-1], 16)

        if keep_addr and target_addr:
            if keep_addr == last_insn_addr:
                return [target_addr]
            return [keep_addr, last_insn_addr, target_addr]

    return None


def patch_ref_to_jmp(bv, ref_addr, target_addr):
    """Apply JMP patch. See stage3_opaque_predicate.py for full documentation."""
    arch = bv.arch

    if len(target_addr) == 1:
        target = target_addr[0]
        try:
            jmp_bytes = arch.assemble(f"jmp {target:#x}", ref_addr)
            original_len = bv.get_instruction_length(ref_addr)
            if len(jmp_bytes) < original_len:
                jmp_bytes = jmp_bytes.ljust(original_len, b'\x90')
            elif len(jmp_bytes) > original_len:
                print(f"  [ERROR] JMP too long ({len(jmp_bytes)} > {original_len})")
                return False
            bv.write(ref_addr, jmp_bytes)
            return True
        except Exception as e:
            print(f"  [ERROR] Patch failed: {e}")
            return False

    elif len(target_addr) == 3:
        keep_addr, jump_addr, target = target_addr
        try:
            jump_keep = arch.assemble(f"jmp {keep_addr:#x}", ref_addr)
            jump_target = arch.assemble(f"jmp {target:#x}", jump_addr)
            original_len = bv.get_instruction_length(ref_addr)
            if len(jump_keep) < original_len:
                jump_keep = jump_keep.ljust(original_len, b'\x90')
            elif len(jump_keep) > original_len:
                return False
            bv.write(ref_addr, jump_keep)
            bv.write(jump_addr, jump_target)
            return True
        except Exception as e:
            print(f"  [ERROR] Patch failed: {e}")
            return False

    return False


# ======================================================================
# Test utilities
# ======================================================================

def test_single_pair(bv, var1_addr, var2_addr, function_name=None):
    """
    Test the deobfuscation algorithm on a single pair of state variables.

    Prints detailed information about each code reference, the resolved
    branch target, and a preview of the patch that would be applied.

    Args:
        bv:            Binary view.
        var1_addr:     Address of the first variable in the pair.
        var2_addr:     Address of the second variable in the pair.
        function_name: Optional function name filter.
    """
    print(f"\n{'=' * 80}")
    print("TESTING SINGLE PAIR")
    print(f"{'=' * 80}\n")
    print(f"Variable 1: {var1_addr:#x}")
    print(f"Variable 2: {var2_addr:#x}")

    if var1_addr not in bv.data_vars or var2_addr not in bv.data_vars:
        print("[ERROR] Variables not found in data_vars")
        return False

    spacing = var2_addr - var1_addr
    print(f"Spacing: {spacing} bytes")
    if spacing != 4:
        print(f"[WARNING] Expected 4-byte spacing, got {spacing}")

    code_refs = list(bv.get_code_refs(var2_addr))
    print(f"\nReferences to {var2_addr:#x}: {len(code_refs)}")

    if not code_refs:
        print("[ERROR] No code references found")
        return False

    if function_name:
        code_refs = [r for r in code_refs if r.function and r.function.name == function_name]
        print(f"After filtering by '{function_name}': {len(code_refs)}")

    for idx, ref in enumerate(code_refs, 1):
        ref_addr = ref.address
        func_name = ref.function.name if ref.function else "Unknown"

        print(f"\n{'─' * 80}")
        print(f"[REF {idx}/{len(code_refs)}] {ref_addr:#x} in {func_name}")
        print(f"{'─' * 80}")

        _print_context(bv, ref_addr)

        target_addr = find_jne_target_from_ref(bv, ref_addr)
        if target_addr:
            if len(target_addr) == 1:
                print(f"\n[TARGET] Resolved: {target_addr[0]:#x}")
            else:
                keep, jump, final = target_addr
                print(f"\n[TARGET] Complex jump:")
                print(f"  Keep:   {keep:#x}")
                print(f"  Jump:   {jump:#x}")
                print(f"  Target: {final:#x}")

            # Show patch preview
            original_bytes = bv.read(ref_addr, bv.get_instruction_length(ref_addr))
            print(f"\n[PATCH PREVIEW]")
            print(f"  Original @ {ref_addr:#x}: {original_bytes.hex()}")
            print(f"  {bv.get_disassembly(ref_addr)}")

            if len(target_addr) == 1:
                try:
                    jmp_bytes = bv.arch.assemble(f"jmp {target_addr[0]:#x}", ref_addr)
                    print(f"  New bytes: {jmp_bytes.hex()}")
                    print(f"  New insn:  jmp {target_addr[0]:#x}")
                except Exception as e:
                    print(f"  [ERROR] Assembly failed: {e}")
        else:
            print("\n[ERROR] Could not resolve target")
            _print_basic_block_info(bv, ref_addr)

    return True


def test_and_apply_patch(bv, var1_addr, var2_addr, function_name=None, dry_run=True):
    """
    Test and optionally apply patches for a specific variable pair.

    Args:
        dry_run: If True, only preview patches without applying them.
    """
    print(f"\n{'=' * 80}")
    print(f"TEST AND {'PREVIEW' if dry_run else 'APPLY'} PATCH")
    print(f"{'=' * 80}\n")

    if not dry_run:
        bv.begin_undo_actions()

    code_refs = list(bv.get_code_refs(var2_addr))
    if function_name:
        code_refs = [r for r in code_refs if r.function and r.function.name == function_name]

    patches_applied = 0
    patches_failed = 0

    for idx, ref in enumerate(code_refs, 1):
        ref_addr = ref.address
        func_name = ref.function.name if ref.function else "Unknown"
        print(f"\n[REF {idx}/{len(code_refs)}] {ref_addr:#x} in {func_name}")

        target_addr = find_jne_target_from_ref(bv, ref_addr)
        if target_addr:
            if not dry_run:
                if patch_ref_to_jmp(bv, ref_addr, target_addr):
                    patches_applied += 1
                else:
                    patches_failed += 1
            else:
                print(f"  [PREVIEW] Would patch: jmp {target_addr}")
                patches_applied += 1
        else:
            print("  [SKIP] No target found")
            patches_failed += 1

    if not dry_run:
        bv.commit_undo_actions()
        if patches_applied > 0:
            print("\nUpdating analysis...")
            bv.update_analysis_and_wait()

    print(f"\n{'=' * 80}")
    print(f"SUMMARY — Mode: {'PREVIEW' if dry_run else 'APPLIED'}")
    print(f"  Patches applied: {patches_applied}")
    print(f"  Patches failed:  {patches_failed}")

    return patches_applied, patches_failed


def find_pairs_in_function(bv, function_name):
    """Find all candidate DWORD pairs referenced within a specific function."""
    func = None
    for f in bv.functions:
        if f.name == function_name:
            func = f
            break

    if not func:
        print(f"[ERROR] Function '{function_name}' not found")
        return []

    print(f"\nSearching for pairs in: {function_name} @ {func.start:#x}")

    data_refs = set()
    for addr in range(func.start, func.start + func.total_bytes):
        data_refs.update(bv.get_data_refs_from(addr))

    sorted_refs = sorted(data_refs)
    pairs = []
    for i in range(len(sorted_refs) - 1):
        if sorted_refs[i + 1] - sorted_refs[i] == 4:
            pairs.append((sorted_refs[i], sorted_refs[i + 1]))

    print(f"Found {len(pairs)} potential pairs")
    for idx, (a1, a2) in enumerate(pairs, 1):
        print(f"  Pair {idx}: {a1:#x}, {a2:#x}")

    return pairs


# ======================================================================
# Internal helpers
# ======================================================================

def _print_context(bv, addr, lines_before=3, lines_after=3):
    """Print disassembly context around an address."""
    func = bv.get_functions_containing(addr)
    if not func:
        return
    func = func[0]
    bb = func.get_basic_block_at(addr)
    if not bb:
        return

    print(f"\nContext at {addr:#x}:")
    print(f"Function: {func.name}")
    print(f"Basic block: {bb.start:#x} - {bb.end:#x}\n")

    instructions = []
    current = bb.start
    while current < bb.end:
        insn_len = bv.get_instruction_length(current)
        if insn_len == 0:
            break
        instructions.append((current, bv.get_disassembly(current)))
        current += insn_len

    target_idx = next((i for i, (a, _) in enumerate(instructions) if a == addr), None)
    if target_idx is None:
        return

    start_idx = max(0, target_idx - lines_before)
    end_idx = min(len(instructions), target_idx + lines_after + 1)

    for idx in range(start_idx, end_idx):
        insn_addr, disasm = instructions[idx]
        marker = ">>>" if insn_addr == addr else "   "
        print(f"{marker} {insn_addr:#x}: {disasm}")


def _print_basic_block_info(bv, addr):
    """Print detailed basic block information for debugging."""
    func = bv.get_functions_containing(addr)
    if not func:
        return
    func = func[0]
    bb = func.get_basic_block_at(addr)
    if not bb:
        return

    print(f"\n[BASIC BLOCK INFO]")
    print(f"  Range: {bb.start:#x} - {bb.end:#x}")
    print(f"  Index: {bb.index}, Length: {bb.length} bytes")

    print(f"  Outgoing edges: {len(bb.outgoing_edges)}")
    for edge in bb.outgoing_edges:
        print(f"    -> {edge.target.start:#x}")

    print(f"  Incoming edges: {len(bb.incoming_edges)}")
    for edge in bb.incoming_edges:
        print(f"    <- {edge.source.start:#x}")

    print(f"  Instructions:")
    current = bb.start
    while current < bb.end:
        insn_len = bv.get_instruction_length(current)
        if insn_len == 0:
            break
        print(f"    {current:#x}: {bv.get_disassembly(current)}")
        current += insn_len
