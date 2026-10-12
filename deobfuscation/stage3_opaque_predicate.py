"""
Stage 3 — Opaque Predicate Deobfuscator (Binary Ninja Plugin)

Removes opaque-predicate obfuscation from the Stage 3 payload.  The
obfuscator introduces pairs of adjacent DWORD global variables that act
as a state machine: a comparison against the second variable of each pair
produces a conditional branch (JNE) whose outcome is predetermined at
compile time.  This script identifies these variable pairs, resolves
the always-taken branch target, and patches the conditional into an
unconditional JMP.

Algorithm:
  1. Scan all data variables to find regions of consecutive DWORD pairs
     that share the same reference count — the signature of the
     obfuscator's state-variable allocation pattern.
  2. For each pair, follow code references to the second variable and
     locate the basic block's terminating JNE instruction.
  3. Walk backward from the JNE to find where the "real" code begins
     (the first non-MOV instruction), separating the state-update
     preamble from the original code.
  4. Patch:
     - Simple case (1 target): replace the reference instruction with
       `jmp <resolved_target>`.
     - Complex case (3 targets): insert a `jmp` to the real-code start
       and a second `jmp` to the resolved branch target.

Environment: Binary Ninja Script Console.
"""

from capstone import Cs, CS_ARCH_X86, CS_MODE_64


def find_continuous_paired_regions(bv):
    """
    Identify contiguous regions of paired DWORD data variables.

    The obfuscator allocates state variables as consecutive DWORD pairs
    (4 bytes apart) where both variables in a pair have the same
    reference count.  This function finds regions containing multiple
    such pairs, which is the fingerprint of the opaque-predicate scheme.
    """
    data_info = []
    for addr in sorted(bv.data_vars.keys()):
        data_var = bv.data_vars[addr]
        ref_count = len(list(bv.get_code_refs(addr)))
        data_info.append({'addr': addr, 'type': str(data_var.type), 'ref_count': ref_count})

    regions = []
    i = 0

    while i < len(data_info) - 1:
        current = data_info[i]
        next_var = data_info[i + 1]

        is_valid_pair = (
            next_var['addr'] - current['addr'] == 4
            and current['ref_count'] == next_var['ref_count']
            and current['ref_count'] > 0
        )

        if is_valid_pair:
            pairs = [(current, next_var)]
            j = i + 2
            consecutive = True

            while j < len(data_info) - 1 and consecutive:
                var1 = data_info[j]
                var2 = data_info[j + 1]
                prev_last = data_info[j - 1]

                if (var1['addr'] - prev_last['addr'] == 4
                        and var2['addr'] - var1['addr'] == 4
                        and var1['ref_count'] == var2['ref_count']
                        and var1['ref_count'] > 0):
                    pairs.append((var1, var2))
                    j += 2
                else:
                    consecutive = False

            if len(pairs) >= 2:
                regions.append({
                    'start_addr': pairs[0][0]['addr'],
                    'end_addr': pairs[-1][1]['addr'],
                    'num_pairs': len(pairs),
                    'pairs': pairs,
                })

            i = j if j > i + 2 else i + 2
        else:
            i += 1

    return regions


def find_jne_target_from_ref(bv, ref_addr):
    """
    Given a code reference to a state variable, locate the JNE target
    address by examining the terminating instruction of the containing
    basic block.

    Returns:
      - [target_addr]                       — simple case (direct jump)
      - [keep_addr, jump_addr, target_addr] — complex case (two jumps needed)
      - None                                — pattern not recognized
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

        # Walk backward to find where the state-update MOV sequence begins
        while True:
            current_addr = bb.get_instruction_containing_address(current_addr - 1)[1]
            current_insn = str(bv.get_disassembly(current_addr))
            if "mov" not in current_insn:
                keep_addr = previous_addr
                break
            previous_addr = current_addr

        # Handle chained JNE blocks (dispatcher chains)
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
        # Entry block with unconditional jump into dispatcher
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
    """
    Patch instruction(s) at ref_addr to unconditional JMP(s) targeting
    the resolved address(es).

    Handles both simple (1 target) and complex (3 targets: keep, jump, target)
    patching layouts.
    """
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
            print(f"  [PATCH] {ref_addr:#x}: jmp {target:#x} ({len(jmp_bytes)} bytes)")
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
                print(f"  [ERROR] JMP too long ({len(jump_keep)} > {original_len})")
                return False
            bv.write(ref_addr, jump_keep)
            bv.write(jump_addr, jump_target)
            print(f"  [PATCH] {ref_addr:#x}: jmp {keep_addr:#x}")
            print(f"  [PATCH] {jump_addr:#x}: jmp {target:#x}")
            return True
        except Exception as e:
            print(f"  [ERROR] Patch failed: {e}")
            return False

    return False


def deobfuscate_opaque_predicates(bv, largest_region):
    """Process all pairs in the largest identified region and apply patches."""
    print(f"\n{'=' * 80}")
    print("DEOBFUSCATION START")
    print(f"{'=' * 80}\n")

    patches_applied = 0
    patches_failed = 0

    for pair_idx, (var1, var2) in enumerate(largest_region['pairs'], 1):
        pair_addr = var2['addr']
        print(f"\n[PAIR {pair_idx}/{largest_region['num_pairs']}] Processing {pair_addr:#x}")

        code_refs = list(bv.get_code_refs(pair_addr))

        for ref_idx, ref in enumerate(code_refs, 1):
            ref_addr = ref.address
            func_name = ref.function.name if ref.function else "Unknown"
            print(f"\n  [REF {ref_idx}/{len(code_refs)}] {ref_addr:#x} in {func_name}")

            target_addr = find_jne_target_from_ref(bv, ref_addr)
            if target_addr:
                if patch_ref_to_jmp(bv, ref_addr, target_addr):
                    patches_applied += 1
                else:
                    patches_failed += 1
            else:
                print("  [SKIP] No target address extracted")
                patches_failed += 1

    print(f"\n{'=' * 80}")
    print("DEOBFUSCATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Patches applied: {patches_applied}")
    print(f"Patches failed:  {patches_failed}")
    print(f"Total pairs:     {largest_region['num_pairs']}")

    return patches_applied


# ======================================================================
# Main execution
# ======================================================================

print("Finding paired regions...")
bv.begin_undo_actions()

regions = find_continuous_paired_regions(bv)
largest = max(regions, key=lambda r: r['num_pairs'])

print(f"\nLargest region: {largest['start_addr']:#x} - {largest['end_addr']:#x}")
print(f"Number of pairs: {largest['num_pairs']}")

patches_count = deobfuscate_opaque_predicates(bv, largest)

if patches_count > 0:
    print("\nUpdating analysis...")
    bv.update_analysis_and_wait()
    print("Analysis updated!")

bv.commit_undo_actions()
print("\n[DONE] Deobfuscation complete!")
